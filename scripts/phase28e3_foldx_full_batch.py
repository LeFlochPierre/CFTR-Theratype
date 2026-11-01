#!/usr/bin/env python3
"""Phase 28E-3 | TRAIN-only full FoldX BuildModel batches (no VTI labels).

Requires completed Phase28E-1, both successful Phase28E-2 pilots, licensed FoldX.
Reuses repaired PDBs, performs immutable 20-substitution batches, validates FoldX
mutation execution logs and report existence. Does NOT interpret energy rows as
mutation-level ΔΔG; that requires a separate, format-verified parsing step.

Default: PLAN ONLY. --execute runs. --resume revalidates completed batch folders,
refuses incomplete folders; never overwrites partial results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

# Import the tested Phase 28E-2 implementation from the scripts folder.
try:
    import phase28e2_foldx_pilot_resume as pilot
except ImportError as e:
    raise SystemExit('Save phase28e2_foldx_pilot_resume.py in the same scripts/ directory first') from e

STRUCTURES = ('7SV7', '8EIG')


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def output_files(work: Path, tag: str, repaired_stem: str) -> dict[str, Path]:
    """Support actual FoldX 5 filename and older documented tag naming."""
    choices = [f'{tag}_{repaired_stem}', f'{tag}_BM']
    selected = []
    for candidate in choices:
        p = work/f'Dif_{candidate}.fxout'
        if p.is_file() and p.stat().st_size > 0:
            selected.append(candidate)
    if len(selected) != 1:
        raise ValueError(f'{work}: expected exactly one difference report, found {selected}; '
                         f'reports={[p.name for p in work.glob("*.fxout")]}')
    suffix = selected[0]
    result = {}
    for role in ('Dif','Average','Raw','PdbList'):
        p = work/f'{role}_{suffix}.fxout'
        if not p.is_file() or p.stat().st_size == 0:
            raise FileNotFoundError(f'{work}: missing/nonempty {p.name}')
        result[role] = p
    return result


def numeric_rows(path: Path) -> tuple[int, list[tuple[str,float]]]:
    # Descriptive only: FoldX 5 pilot produced 16 numeric lines for 5*3 runs.
    # Do NOT equate raw numeric lines with unique mutation-run rows.
    vals=[]
    for line in path.read_text(errors='replace').splitlines():
        parts=line.split('\t')
        if len(parts)<2:
            continue
        try:
            v=float(parts[1])
        except (ValueError, TypeError):
            continue
        if math.isfinite(v):
            vals.append((parts[0],v))
    return len(vals), vals


def plan_inputs(source: Path, pilot_root: Path, batch_size: int, only: str):
    if not (1 <= batch_size <= 50):
        raise ValueError('--batch-size must be between 1 and 50')
    original, prov=pilot.validate_sources(source)
    selected = STRUCTURES if only == 'both' else (only,)
    tasks=[]
    for name in selected:
        unrep, full, _ = original[name]
        work = pilot_root/name
        if not work.is_dir():
            raise FileNotFoundError(f'{name} pilot workspace missing: {work}')
        pilot_qc=work/'pilot_qc.json'
        if not pilot_qc.is_file():
            raise FileNotFoundError(f'{name} pilot summary missing: {pilot_qc}')
        repaired=work/(unrep.stem+'_Repair.pdb')
        # Strongly verify repaired site identities and original reference coverage.
        repaired_qc = pilot.validate_repair(unrep, repaired, full)
        old = json.loads(pilot_qc.read_text())
        if old.get('input_sha256')!=sha(unrep):
            raise ValueError(f'{name}: E2 pilot provenance disagrees with Phase28E-1 template')
        if old.get('repair',{}).get('repaired_sha256') != sha(repaired):
            raise ValueError(f'{name}: repaired structure changed since successful pilot')
        # Verify presence of actual pilot energy outputs before admitting full calculation.
        pilot.check_diff(work, f'{name}_PILOT', repaired)
        mutations=full
        if len(set(mutations))!=len(mutations):
            raise ValueError(f'{name}: duplicate protein substitutions in original list')
        chunks=[mutations[i:i+batch_size] for i in range(0,len(mutations),batch_size)]
        tasks.append(dict(name=name, unrep=unrep, repaired=repaired, qc=repaired_qc,
                          mutations=mutations, chunks=chunks))
    return tasks,prov


def validate_batch(work: Path, repaired: Path, expected: list[str], runs: int,
                   tag: str, foldx_sha: str, manifest_sha: str) -> dict:
    state_file=work/'batch_manifest.json'
    if not state_file.is_file():
        raise RuntimeError(f'{work}: incomplete workspace (no batch_manifest.json); refusing overwrite')
    state=json.loads(state_file.read_text())
    for key,value in {'mutations':expected,'runs':runs,'tag':tag,
                      'foldx_sha256':foldx_sha,'source_manifest_sha256':manifest_sha,
                      'repaired_sha256':sha(repaired)}.items():
        if state.get(key)!=value:
            raise ValueError(f'{work}: manifest mismatch for {key}')
    local=work/repaired.name
    if not local.is_file() or sha(local)!=sha(repaired):
        raise ValueError(f'{work}: repaired PDB differs from audited pilot')
    mutation_text='\n'.join(expected)+'\n'
    if (work/'individual_list.txt').read_text()!=mutation_text:
        raise ValueError(f'{work}: mutation input has changed')
    log=work/'buildmodel_console.log'
    if not log.is_file():
        raise RuntimeError(f'{work}: BuildModel log absent; refusing overwrite')
    t=log.read_text(errors='replace')
    if 'RETURN_CODE: 0' not in t or 'Your file run OK' not in t:
        raise RuntimeError(f'{work}: BuildModel not confirmed successful; inspect log')
    for part in (f'--numberOfRuns={runs}', f'--output-file={tag}',f'--pdb={repaired.name}'):
        if part not in t:
            raise ValueError(f'{work}: missing expected archived command: {part}')
    # Prior real pilot had 1 Reading per substitution, and exactly 3 Mutating per substitution.
    reading = re.findall(r'(?m)^Reading\s+([A-Z]A\d+[A-Z];)\s*$', t)
    if reading!=expected:
        raise ValueError(f'{work}: FoldX Reading mutations did not match expected list/order '
                         f'(expected={len(expected)}, got={len(reading)})')
    mutation_calls=len(re.findall(r'(?m)^Mutating residue\s*=', t))
    if mutation_calls!=len(expected)*runs:
        raise ValueError(f'{work}: BuildModel mutation calls={mutation_calls}, '
                         f'expected {len(expected)*runs}')
    files=output_files(work,tag,repaired.stem)
    count, vals = numeric_rows(files['Dif'])
    # This is an integrity sanity check only; need a separate schema-aware parser
    # before assigning one ΔΔG per mutation. Real pilot had 16 vs expected 15.
    if count < len(expected)*runs:
        raise ValueError(f'{work}: numeric energy rows={count} fewer than '
                         f'expected mutation-run count={len(expected)*runs}')
    energy_sample=[{'name':name,'column2':value} for name,value in vals[:3]]
    return {'n_mutations':len(expected),'n_runs':runs,'n_mutation_calls':mutation_calls,
            'n_reading_lines':len(reading), 'numeric_energy_rows_unparsed':count,
            'extra_numeric_rows_unexplained':count-len(expected)*runs,
            'energy_first_rows_unparsed':energy_sample,
            'files':{kind:{'path':str(p.resolve()),'sha256':sha(p),'size':p.stat().st_size}
                     for kind,p in files.items()}}


def run_logged(cmd,work:Path,log:Path,timeout_minutes):
    print('  RUN:', ' '.join(cmd),flush=True)
    try:
        p=subprocess.run(cmd,cwd=work,text=True,capture_output=True,timeout=timeout_minutes*60)
    except subprocess.TimeoutExpired as e:
        log.write_text(f'TIMEOUT after {timeout_minutes} minutes; outputs may be partial\n')
        raise RuntimeError(f'FoldX timeout: {work}. Do not overwrite partial results.') from e
    log.write_text('COMMAND: '+' '.join(cmd)+'\nRETURN_CODE: '+str(p.returncode)
                   +'\nSTDOUT:\n'+p.stdout+'\nSTDERR:\n'+p.stderr)
    if p.returncode:
        raise RuntimeError(f'{work}: FoldX failed exit={p.returncode}; inspect {log}')


def self_test():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        w=Path(tmp); r=w/'CFTR_X_Repair.pdb'; r.write_text('test')
        suffix='X_BATCH_001_CFTR_X_Repair'
        for role in ('Dif','Average','Raw','PdbList'):
            (w/f'{role}_{suffix}.fxout').write_text('Pdb\tTotal Energy\nx\t-1.5\n')
        got=output_files(w,'X_BATCH_001',r.stem)
        assert len(got)==4
        n,v=numeric_rows(got['Dif'])
        assert n==1 and v[0][1]==-1.5
        assert sha(got['Dif'])
    print('PASS: report naming, file integrity and numeric-column scanner')


def main(a):
    self_test()
    src=Path(a.source).resolve(); pilot_root=Path(a.pilot_root).resolve()
    out=Path(a.out).resolve()
    tasks,prov=plan_inputs(src,pilot_root,a.batch_size,a.only)
    exe=pilot.binary_path(a.foldx)
    if a.execute and not exe:
        raise FileNotFoundError('Specify installed, licensed FoldX with --foldx')
    exe_sha=sha(Path(exe)) if exe else None
    manifest_sha=prov['inputs']['manifest_sha256']
    print('='*106)
    print('PHASE 28E-3 | FULL FOLDX MUTATION STABILITY CALCULATIONS | NO VTI LABELS')
    print('='*106)
    print(f'Mode={"EXECUTE" if a.execute else "PLAN ONLY"} | --resume={a.resume} | runs={a.runs} '
          f'| batch size={a.batch_size} | FoldX={exe or "not found"}')
    print(f'Immutable Phase28E-1 manifest SHA256: {manifest_sha}')
    rows=[]
    for t in tasks:
        name=t['name']; rep=t['repaired']
        print(f'\n{name}: {len(t["mutations"])} substitutions, {len(t["chunks"])} independent batches')
        print(f'  Repair reused: {rep} | SHA256 {sha(rep)[:16]}...')
        for idx, chunk in enumerate(t['chunks'],1):
            tag=f'{name}_FULL_{idx:03d}'
            w=out/name/f'batch_{idx:03d}'
            print(f'  [{idx:02d}/{len(t["chunks"]):02d}] {len(chunk):2d} mutations '
                  f'{chunk[0]} ... {chunk[-1]}', flush=True)
            if not a.execute:
                continue
            if w.exists() and any(w.iterdir()):
                if not a.resume:
                    raise FileExistsError(f'{w}: exists. Use --resume to verify; never overwrite.')
                qc=validate_batch(w,rep,chunk,a.runs,tag,exe_sha,manifest_sha)
                print(f'    RESUMED: verified existing batch; {qc["n_mutation_calls"]} model calls',flush=True)
            else:
                w.mkdir(parents=True,exist_ok=True)
                shutil.copy2(rep,w/rep.name)
                (w/'individual_list.txt').write_text('\n'.join(chunk)+'\n')
                state={'structure':name,'batch':idx,'mutations':chunk,'runs':a.runs,
                       'tag':tag,'foldx_sha256':exe_sha,'source_manifest_sha256':manifest_sha,
                       'repaired_sha256':sha(rep)}
                (w/'batch_manifest.json').write_text(json.dumps(state,indent=2)+'\n')
                cmd=[exe,'--command=BuildModel',f'--pdb={rep.name}',
                     '--mutant-file=individual_list.txt',f'--numberOfRuns={a.runs}',
                     '--out-pdb=false',f'--output-file={tag}']
                run_logged(cmd,w,w/'buildmodel_console.log',a.timeout_minutes)
                qc=validate_batch(w,rep,chunk,a.runs,tag,exe_sha,manifest_sha)
                print(f'    PASS: mutation calls={qc["n_mutation_calls"]}; '
                      f'raw numeric rows={qc["numeric_energy_rows_unparsed"]} '
                      '(ΔΔG not yet assigned to mutations)',flush=True)
            (w/'batch_qc.json').write_text(json.dumps(qc,indent=2)+'\n')
            rows.append({'pdb':name,'batch':idx,'mutations':len(chunk),'runs':a.runs,
                         'foldx_calls':qc['n_mutation_calls'],
                         'numeric_rows_unparsed':qc['numeric_energy_rows_unparsed'],
                         'extra_numeric_rows':qc['extra_numeric_rows_unexplained'],
                         'diff_sha256':qc['files']['Dif']['sha256'],
                         'batch_dir':str(w)})
    if not a.execute:
        print('\nPLAN COMPLETE: nothing calculated, no files modified.')
        return
    out.mkdir(parents=True,exist_ok=True)
    summary=pd.DataFrame(rows)
    summary.to_csv(out/'phase28e3_batch_summary.csv',index=False)
    for name in [t['name'] for t in tasks]:
        n=summary[summary.pdb==name]['mutations'].sum()
        expected=next(t for t in tasks if t['name']==name)
        if n!=len(expected['mutations']):
            raise RuntimeError(f'{name}: incomplete mutation coverage!')
        print(f'\n{name}: PASS {n}/{len(expected["mutations"])} mutation inputs; '
              f'{n*a.runs} mutation calls; {len(summary[summary.pdb==name])} batches')
    print('\nPASS: full FoldX BuildModel runs complete, all log counts and file sets verified.')
    print('Next: verify detailed fxout row identifiers / replication and derive mutation-level ΔΔG.')
    print('No VTI labels read, no model fit, and no energy-response associations computed.')
    print('Results:', out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default='results/biology/phase28e')
    p.add_argument('--pilot-root',default='results/biology/phase28e/phase28e2_pilot')
    p.add_argument('--out',default='results/biology/phase28e/phase28e3_full')
    p.add_argument('--foldx',default=None)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--runs',type=int,default=3)
    p.add_argument('--batch-size',type=int,default=20)
    p.add_argument('--timeout-minutes',type=int,default=120)
    p.add_argument('--only',choices=('both',*STRUCTURES),default='both')
    args=p.parse_args()
    if args.self_test:self_test()
    else:main(args)
