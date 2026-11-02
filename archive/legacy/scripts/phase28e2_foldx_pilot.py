#!/usr/bin/env python3
"""Phase 28E-2: FoldX protein-only CFTR repair + mutation pilot, no VTI labels.

Requires Phase28E-1 outputs and a licensed FoldX executable. By default this
prepares and prints a non-executing plan; add --execute to perform RepairPDB
and a *small*, domain-balanced BuildModel batch. Do not use this to calculate
or claim VTI treatment effects. The ligand-stripped, drug-bound, engineered
CFTR structures are template-dependent biophysical approximations.

Inputs are mutation identifiers and E1 provenance only; no VTI labels or
model outcomes are loaded. Separate template workspaces to isolate outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

STRUCTURES = ('7SV7', '8EIG')
AA = 'ACDEFGHIKLMNPQRSTVWY'
MUT = re.compile(rf'^([{AA}])A(\d{{1,4}})([{AA}]);$')
MAJOR_DOMAINS = ('TMD1', 'NBD1', 'TMD2', 'NBD2', 'other')


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def structure_sites(path: Path) -> dict[int, tuple[str, bool]]:
    model = PDBParser(QUIET=True).get_structure(path.stem, str(path))[0]
    if list(model.child_dict) != ['A']:
        raise ValueError(f'{path}: expected only protein chain A; found {list(model.child_dict)}')
    sites = {}
    for r in model['A']:
        if r.id[0] != ' ':
            continue
        if r.id[2] != ' ':
            raise ValueError(f'{path}: unexpected residue insertion code {r.id}')
        p = int(r.id[1])
        if p in sites:
            raise ValueError(f'{path}: duplicated residue number {p}')
        letter = seq1(r.resname, custom_map={'UNK': 'X'})
        sites[p] = (letter, all(x in r for x in ('N','CA','C','O')))
    return sites


def validate_repair(input_pdb: Path, repaired_pdb: Path, substitutions: list[str]) -> dict:
    orig, repair = structure_sites(input_pdb), structure_sites(repaired_pdb)
    if orig.keys() != repair.keys():
        lost = sorted(orig.keys() - repair.keys())
        gained = sorted(repair.keys() - orig.keys())
        raise ValueError(f'{repaired_pdb}: repaired residue numbering differs; missing={lost[:12]} added={gained[:12]}')
    changes = [(p, orig[p][0], repair[p][0]) for p in orig if orig[p][0] != repair[p][0]]
    if changes:
        raise ValueError(f'{repaired_pdb}: repaired template changes amino-acid identity: {changes[:10]}')
    for code in substitutions:
        m = MUT.fullmatch(code)
        if m is None:
            raise ValueError(f'Invalid FoldX mutation code {code!r}')
        w,p,alt = m.group(1),int(m.group(2)),m.group(3)
        if p not in repair or repair[p] != (w, True):
            raise ValueError(f'FoldX pilot site {code} is missing, mismatched or has incomplete backbone: {repair.get(p)}')
    return {'site_count': len(orig), 'identical_residue_numbering': True,
            'identical_residue_identities': True,
            'repaired_sha256':sha(repaired_pdb)}


def pick_pilot(eligible: pd.DataFrame, max_per_domain: int = 1) -> pd.DataFrame:
    """Deterministically pick by variant ID, independent of treatment response."""
    d = eligible.drop_duplicates(['pdb', 'mutation']).copy()
    d['protein_position'] = pd.to_numeric(d.protein_position, errors='raise').astype(int)
    # Pick the residue nearest the median within each domain to avoid bias to
    # the N-terminus; no outcome data are read, nor used for selection.
    picks = []
    for domain in MAJOR_DOMAINS:
        part = d[d.domain == domain].copy()
        if part.empty: continue
        center = part.protein_position.median()
        part['distance_from_domain_median'] = (part.protein_position - center).abs()
        part = part.sort_values(['distance_from_domain_median','protein_position','mutation'])
        picks.append(part.head(max_per_domain))
    if not picks:
        raise ValueError('No eligible mutations available for pilot')
    return pd.concat(picks, ignore_index=True)


def validate_sources(source: Path):
    prov_path = source/'phase28e1_provenance.json'
    manifest_path = source/'phase28e1_foldx_mutation_manifest.csv'
    if not prov_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError('Phase28E-1 manifest/provenance missing; run Phase28E-1 first')
    prov = json.loads(prov_path.read_text())
    if prov.get('inputs', {}).get('manifest_sha256') != sha(manifest_path):
        raise ValueError('Phase28E-1 manifest SHA256 has changed')
    m = pd.read_csv(manifest_path, dtype={'variant_key':str, 'mutation':str,'foldx_mutation':str}, keep_default_na=False)
    required = {'pdb','variant_key','mutation','protein_position','domain','eligible','foldx_mutation'}
    if not required.issubset(m.columns) or len(m)!=420 or m.variant_key.nunique()!=210:
        raise ValueError('Unexpected Phase28E-1 manifest schema/cardinality')
    if set(m.pdb.unique()) != set(STRUCTURES):
        raise ValueError('Unexpected structure names in manifest')
    result = {}
    for name in STRUCTURES:
        d=m[(m.pdb==name) & (m.eligible==1)]
        input_pdb=source/name/f'CFTR_{name}_chainA_protein_only.pdb'
        full_mut=source/name/'individual_list.txt'
        if not input_pdb.is_file() or not full_mut.is_file():
            raise FileNotFoundError(f'Missing Phase28E-1 template or mutation list for {name}')
        if sha(input_pdb)!=prov[name]['protein_only_pdb_sha256']:
            raise ValueError(f'{name}: prepared template SHA mismatch')
        lines=[x.strip() for x in full_mut.read_text().splitlines() if x.strip()]
        expected = d.foldx_mutation.drop_duplicates().tolist()
        if len(lines)!=len(expected) or lines!=expected:
            raise ValueError(f'{name}: prepared mutation list differs from Phase28E-1 manifest')
        initial=structure_sites(input_pdb)
        for code in expected:
            mt=MUT.fullmatch(code)
            if mt is None or initial.get(int(mt.group(2))) != (mt.group(1),True):
                raise ValueError(f'{name}: mutation {code} does not match source structure')
        if initial.get(1371,(None,None))[0]!='Q':
            raise ValueError(f'{name}: engineered E1371Q site not retained')
        if name=='8EIG' and 508 in initial:
            raise ValueError('8EIG F508del reference site unexpectedly present')
        pilot=pick_pilot(d)
        result[name] = (input_pdb, expected, pilot)
    return result, prov


def binary_path(requested: str | None) -> str | None:
    if requested:
        if Path(requested).is_file():
            if not os.access(requested,os.X_OK):
                raise PermissionError(f'FoldX executable not executable: {requested}')
            return str(Path(requested).resolve())
        match=shutil.which(requested)
        if not match: raise FileNotFoundError(f'FoldX executable not found: {requested}')
        return str(Path(match).resolve())
    for name in ('foldx', 'FoldX','foldx5'):
        path=shutil.which(name)
        if path: return str(Path(path).resolve())
    return None


def run_logged(cmd: list[str], cwd: Path, log: Path, timeout_minutes: int):
    print('CMD:', ' '.join(shlex.quote(c) for c in cmd),flush=True)
    try:
        result=subprocess.run(cmd,cwd=cwd,capture_output=True,text=True,timeout=timeout_minutes*60)
    except subprocess.TimeoutExpired as e:
        log.write_text(f'TIMEOUT after {timeout_minutes} minutes; output may be partial\n')
        raise RuntimeError(f'FoldX timed out; inspect {cwd}') from e
    log.write_text('COMMAND: '+' '.join(shlex.quote(c) for c in cmd)+'\n'
                   +f'RETURN_CODE: {result.returncode}\nSTDOUT:\n'+result.stdout+'\nSTDERR:\n'+result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f'FoldX returned {result.returncode}; inspect {log}')
    print(f'PASS command exit=0; log={log}',flush=True)


def check_diff(run_path: Path, tag: str) -> dict:
    diff=run_path/f'Dif_{tag}_BM.fxout'
    if not diff.is_file() or diff.stat().st_size==0:
        raise FileNotFoundError(f'Expected FoldX difference-energy report missing/empty: {diff}')
    print('  Difference-energy report:',diff)
    # Parse only format-independent basic diagnostics here. Full row-to-variant
    # matching and energy interpretation belong in a dedicated Phase28E-3.
    lines=diff.read_text(errors='replace').splitlines()
    print('  Header/sample lines:')
    for line in lines[:min(6,len(lines))]:
        print('   ',line[:210])
    if len(lines)<2:
        raise ValueError(f'{diff}: too few lines to contain meaningful energy results')
    return {'diff_fxout':str(diff.resolve()),'diff_sha256':sha(diff),
            'diff_bytes':diff.stat().st_size,'diff_lines':len(lines)}


def self_test():
    d=pd.DataFrame([
        {'pdb':'7SV7','mutation':'A90C','protein_position':90,'domain':'TMD1'},
        {'pdb':'7SV7','mutation':'A110C','protein_position':110,'domain':'TMD1'},
        {'pdb':'7SV7','mutation':'I444T','protein_position':444,'domain':'NBD1'},
        {'pdb':'7SV7','mutation':'L1091F','protein_position':1091,'domain':'TMD2'},
        {'pdb':'7SV7','mutation':'P1290L','protein_position':1290,'domain':'NBD2'}])
    p=pick_pilot(d)
    assert len(p)==4 and set(p.domain)=={'TMD1','NBD1','TMD2','NBD2'}
    assert MUT.fullmatch('IA203T;') and MUT.fullmatch('IA203T') is None
    print('PASS: deterministic mutation selection, FoldX syntax and quality gates')


def main(a):
    self_test()
    source=Path(a.source).resolve(); out=Path(a.out).resolve()
    inputs,prov=validate_sources(source)
    foldx=binary_path(a.foldx)
    print('='*108)
    print('PHASE 28E-2 | FOLDX REPAIR + LIMITED PILOT | NO VTI OUTCOMES')
    print('='*108)
    print('FoldX executable:',foldx or 'NOT FOUND (dry-run plan only)')
    print('Mode:', 'EXECUTE' if a.execute else 'PLAN ONLY')
    print('BuildModel repeats:',a.runs)
    print('Input manifest hash:',prov['inputs']['manifest_sha256'])
    if a.execute and foldx is None:
        raise FileNotFoundError('FoldX not installed/on PATH. Obtain licensed FoldX, pass --foldx /full/path/to/binary')
    if a.runs not in (1,2,3,4,5,10):
        raise ValueError('Use --runs 1,2,3,4,5 or 10')
    report={'phase':'28E-2','executed':a.execute,'runs':a.runs,'foldx':foldx,
            'source_manifest_sha256':prov['inputs']['manifest_sha256'],'templates':{}}
    for name in STRUCTURES:
        pdb,full,pilot=inputs[name]
        entries=pilot[['mutation','foldx_mutation','domain','protein_position']].drop_duplicates().copy()
        if entries.foldx_mutation.duplicated().any():
            raise ValueError(f'{name}: repeated pilot substitutions')
        print(f'\n{name}: {len(full)} eligible substitutions; pilot {len(entries)}:',
              ', '.join(entries.mutation.tolist()))
        print('  Source PDB:',pdb)
        work=out/name
        staged_pdb=work/pdb.name
        repaired=work/(pdb.stem+'_Repair.pdb')
        if not a.execute:
            print('  Would repair:',staged_pdb.name)
            print('  Would verify residue numbering and all eligible sites')
            print(f'  Would run BuildModel on {len(entries)} domain-balanced substitutions ({a.runs} repeats)')
            continue
        if work.exists() and any(work.iterdir()):
            raise FileExistsError(f'{work} is nonempty. Refusing to overwrite; use a fresh --out directory')
        work.mkdir(parents=True,exist_ok=True)
        shutil.copy2(pdb,staged_pdb)
        entries.to_csv(work/'pilot_selected_mutations.csv',index=False)
        (work/'individual_list_pilot.txt').write_text('\n'.join(entries.foldx_mutation)+'\n')
        # FoldX's mutant-file must be named individual_list*.txt for some versions.
        (work/'individual_list.txt').write_text('\n'.join(entries.foldx_mutation)+'\n')
        cmd=[foldx,'--command=RepairPDB',f'--pdb={staged_pdb.name}']
        run_logged(cmd,work,work/'repair_console.log',a.timeout_minutes)
        if not repaired.is_file():
            raise FileNotFoundError(f'{name}: RepairPDB did not produce expected {repaired.name}; inspect repair_console.log')
        qc=validate_repair(staged_pdb,repaired,full)
        print('  Repair QC: PASS; sites=',qc['site_count'],'all eligible mutations remap')
        tag=f'{name}_PILOT'
        cmd=[foldx,'--command=BuildModel',f'--pdb={repaired.name}',
             '--mutant-file=individual_list.txt',f'--numberOfRuns={a.runs}',
             '--out-pdb=false',f'--output-file={tag}']
        run_logged(cmd,work,work/'buildmodel_console.log',a.timeout_minutes)
        result=check_diff(work,tag)
        report['templates'][name]={'pilot_mutations':entries.mutation.tolist(),
                                    'input_sha256':sha(pdb),'repair':qc,'result':result}
        (work/'pilot_qc.json').write_text(json.dumps(report['templates'][name],indent=2)+'\n')
    if a.execute:
        out.mkdir(parents=True,exist_ok=True)
        (out/'phase28e2_pilot_summary.json').write_text(json.dumps(report,indent=2)+'\n')
        print('\nPASS Phase 28E-2: FoldX pilot completed and numbering verified for both structures.')
        print('Do not interpret FoldX Dif energies as VTI rescue or as direct drug-response effects.')
    else:
        print('\nPLAN COMPLETE: Nothing executed or modified. Install licensed FoldX and rerun with --execute.')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--source',default='results/biology/phase28e',help='Prepared Phase 28E-1 directory')
    ap.add_argument('--out',default='results/biology/phase28e/phase28e2_pilot',help='Fresh output directory')
    ap.add_argument('--foldx',default=None,help='Licensed FoldX executable absolute path or PATH name')
    ap.add_argument('--execute',action='store_true',help='Run licensed FoldX; default is dry-run plan')
    ap.add_argument('--runs',type=int,default=3)
    ap.add_argument('--timeout-minutes',type=int,default=120)
    args=ap.parse_args()
    if args.self_test:self_test()
    else:main(args)
