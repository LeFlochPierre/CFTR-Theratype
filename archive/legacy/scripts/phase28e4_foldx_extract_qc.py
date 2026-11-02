#!/usr/bin/env python3
"""Phase 28E-4 | read-only FoldX mutation energy extraction and strict mapping QC.

This script NEVER reads VTI outcomes or fits models. It verifies Phase 28E-3
batch provenance, extracts per-run FoldX difference energies, matches them to
mutation input lines, and reports uncertainty and cross-template agreement.

Do not treat structural ΔΔG as measured CFTR unfolding energy or VTI response.
If an unknown FoldX row naming scheme is encountered, the script fails closed,
printing the first few report rows for diagnosis.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AA = set('ACDEFGHIKLMNPQRSTVWY')
MT = re.compile(r'^([ACDEFGHIKLMNPQRSTVWY])A(\d{1,4})([ACDEFGHIKLMNPQRSTVWY]);$')


def sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()


def report_paths(w: Path, tag: str, repaired_name: str) -> dict[str,Path]:
    stem=Path(repaired_name).stem
    suffixes=[f'{tag}_{stem}', f'{tag}_BM']
    matches=[s for s in suffixes if (w/f'Dif_{s}.fxout').is_file()]
    if len(matches)!=1: raise ValueError(f'{w}: expected unique Dif report, found {matches}')
    s=matches[0]; out={k:w/f'{k}_{s}.fxout' for k in ('Dif','Raw','Average','PdbList')}
    for kind,p in out.items():
        if not p.is_file() or not p.stat().st_size: raise ValueError(f'{w}: missing/empty {kind}: {p}')
    return out


def table_lines(path:Path):
    return [line.strip('\r\n') for line in path.read_text(errors='replace').splitlines() if line.strip()]


def parse_dif(path:Path):
    """Read only actual FoldX energy data: second column and numeric decomposition.

    Reject metadata with too few energy columns, including numeric pilot footer.
    """
    rows=[]; headers=[]
    for line in table_lines(path):
        p=[x.strip() for x in line.split('\t')]
        if p and p[0].lower()=='pdb' and len(p)>1 and 'energy' in p[1].lower():
            headers.append(p)
            continue
        if len(p)<5 or not p[0] or p[0].startswith('#'): continue
        try: vals=[float(x) for x in p[1:]]
        except ValueError: continue
        if len(vals)<4: continue
        if not all(math.isfinite(x) for x in vals):
            raise ValueError(f'{path}: non-finite energy value: {line[:180]}')
        rows.append({'model':p[0], 'ddg_kcal_mol':vals[0], 'energy_columns':len(vals)})
    if not rows: raise ValueError(f'{path}: no full FoldX energy rows: '+str(table_lines(path)[:12]))
    if len(set(x['model'] for x in rows))!=len(rows):
        raise ValueError(f'{path}: repeated Pdb IDs in Dif_ data rows')
    if headers:
        h=headers[-1]
        if 'total' not in h[1].lower():
            raise ValueError(f'{path}: ambiguous energy units/column: {h[:3]}')
    else:
        raise ValueError(f'{path}: missing Pdb / Total Energy column header')
    return rows,headers[-1]


def model_tokens(path: Path):
    """Retrieve PDB identifiers from PdbList lines (not from guessed assignments)."""
    vals=[]
    for line in table_lines(path):
        tokens=re.findall(r'[^\s,;]+?\.pdb',line,flags=re.I)
        for t in tokens:
            name=Path(t.strip('"\'')).name
            if name.lower().startswith('wt_') or name.lower()=='pdb': continue
            vals.append(name)
    return vals


def canonical_id(x):
    return Path(str(x)).name.strip().removesuffix('.pdb').removesuffix('.PDB').lower()


def map_rows(rows: list[dict], expected: list[str], runs: int, pdb_list: Path):
    n=len(expected)
    if len(rows)!=n*runs:
        raise ValueError(f'{pdb_list}: Dif energy rows {len(rows)} != {n} mutations x {runs} runs')
    names=[canonical_id(x['model']) for x in rows]
    ids=model_tokens(pdb_list)
    names_from_list=[canonical_id(x) for x in ids]
    if not names_from_list:
        raise ValueError(f'{pdb_list}: no PdbList model names found; refuse unverified mapping')
    # Strict cross-file identity and, if possible, order agreement.
    if sorted(names)!=sorted(names_from_list):
        raise ValueError(f'{pdb_list}: PdbList model IDs do not match Dif IDs; '
                         f'Dif first={names[:4]}; PdbList first={names_from_list[:4]}')
    if names!=names_from_list:
        raise ValueError(f'{pdb_list}: PdbList and Dif order differ; refusing order-based mapping')
    # Prefer explicit FoldX ID (model mutation ordinal + replicate index).
    # Typical IDs end ..._1_0, ..._1_1, ..._1_2, ..._2_0, etc.
    suffixes=[re.search(r'_(\d+)_(\d+)$',x) for x in names]
    assignments=[]
    if all(x is not None for x in suffixes):
        pairs=[(int(m.group(1)),int(m.group(2))) for m in suffixes]
        unique_ord=sorted(set(k for k,r in pairs))
        run_numbers=sorted(set(r for k,r in pairs))
        # Match 1-based or 0-based mutation indices; run indices can be either.
        if unique_ord not in (list(range(1,n+1)),list(range(n))):
            raise ValueError(f'{pdb_list}: unexpected mutation ordinal suffixes {unique_ord[:8]}, n={n}')
        if run_numbers not in (list(range(runs)), list(range(1,runs+1))):
            raise ValueError(f'{pdb_list}: unexpected replicate suffixes {run_numbers}; expected {runs}')
        base=1 if unique_ord[0]==1 else 0
        if set(pairs) != {(i+base,r) for i in range(n) for r in run_numbers}:
            raise ValueError(f'{pdb_list}: missing/duplicated mutation × repeat IDs')
        for row,(ordn,run) in zip(rows,pairs):
            i=ordn-base
            assignments.append(dict(foldx_mutation=expected[i], mutation_index=i, replicate=run,
                                    model=row['model'],ddg_kcal_mol=row['ddg_kcal_mol'],
                                    mapping_method='pdb_id_two_indices_crosschecked'))
    else:
        # Some FoldX versions enumerate individual models with a single ordinal.
        # Accept only an explicit 1..N*runs run-major index, with PdbList order
        # matching Dif, and only with --allow-sequential-models (opt-in).
        raise ValueError(
            f'{pdb_list}: FoldX model IDs lack explicit mutation/repeat indices. '
            f'First IDs: {names[:8]}. Refusing to guess assignment; inspect report samples.')
    return assignments


def parse_code(x):
    m=MT.fullmatch(x)
    if m is None: raise ValueError(f'Invalid FoldX code: {x}')
    return f'{m.group(1)}{int(m.group(2))}{m.group(3)}',int(m.group(2))


def inspect_batch(w:Path,tag:str,repaired:str):
    reports=report_paths(w,tag,repaired)
    print('  Report:', reports['Dif'])
    for k in ('Dif','Average','PdbList'):
        print(f'  [{k}] sample:')
        lines=table_lines(reports[k]); hits=[z for z in lines if '\t' in z and ('.pdb' in z.lower() or 'total energy' in z.lower())]
        for z in (hits[:7] if hits else lines[:9]): print('    '+z[:240])


def load_manifest(root:Path):
    p=root/'phase28e1_foldx_mutation_manifest.csv'
    prov=root/'phase28e1_provenance.json'
    if not p.is_file() or not prov.is_file(): raise FileNotFoundError('Missing immutable Phase 28E-1 manifest/provenance')
    j=json.loads(prov.read_text())
    if sha(p)!=j['inputs']['manifest_sha256']: raise ValueError('Phase 28E-1 manifest hash changed')
    d=pd.read_csv(p,keep_default_na=False)
    return d,j


def extract(root:Path,full:Path,out:Path,inspect_only:bool):
    manifest,provenance=load_manifest(root)
    paths=sorted(full.glob('*/batch_???'))
    if len(paths)!=20: raise ValueError(f'Expected 20 full batches, found {len(paths)}')
    all_rows=[]; jobs=[]
    print('='*104)
    print('PHASE 28E-4 | FOLDX ENERGY EXTRACTION + ID/REPLICATE QC | NO VTI OUTCOMES')
    print('='*104)
    print('Mode:', 'INSPECT (read-only)' if inspect_only else 'EXTRACT (read outputs; write validated CSVs)')
    for s in ('7SV7','8EIG'):
        subpaths=sorted(x for x in paths if x.parent.name==s)
        if len(subpaths)!=10: raise ValueError(f'{s}: expected 10 batches; got {len(subpaths)}')
        for w in subpaths:
            state=json.loads((w/'batch_manifest.json').read_text())
            old=json.loads((w/'batch_qc.json').read_text())
            runs=int(state['runs']); exp=state['mutations']; tag=state['tag']
            if runs!=3: raise ValueError(f'{w}: unexpected replicate count {runs}')
            if state['source_manifest_sha256']!=provenance['inputs']['manifest_sha256']:
                raise ValueError(f'{w}: E1 manifest SHA mismatch')
            repaired=next(w.glob('*_Repair.pdb'),None)
            if repaired is None or sha(repaired)!=state['repaired_sha256']:
                raise ValueError(f'{w}: repaired PDB hash mismatch')
            if (w/'individual_list.txt').read_text()!='\n'.join(exp)+'\n':
                raise ValueError(f'{w}: mutated individual_list')
            files=report_paths(w,tag,repaired.name)
            for role,p in files.items():
                if sha(p)!=old['files'][role]['sha256']:
                    raise ValueError(f'{w}: {role} SHA mismatch vs batch QC')
            if inspect_only:
                if w.name=='batch_001': inspect_batch(w,tag,repaired.name)
                continue
            rows,hdr=parse_dif(files['Dif'])
            try: mapped=map_rows(rows,exp,runs,files['PdbList'])
            except Exception as exc:
                print(f'\nFAILED mapping at {w} — no output CSV written. Reason: {exc}')
                inspect_batch(w,tag,repaired.name)
                raise
            if len(mapped)!=len(exp)*runs: raise ValueError(f'{w}: wrong mapped count')
            for m in mapped:
                mutation,pos=parse_code(m['foldx_mutation'])
                m.update(structure=s,batch=w.name,mutation=mutation,protein_position=pos,
                         diff_sha256=sha(files['Dif']),repaired_sha256=state['repaired_sha256'])
            all_rows.extend(mapped)
            jobs.append(dict(structure=s,batch=w.name,mutations=len(exp),energy_rows=len(mapped),
                             diff_sha256=sha(files['Dif'])))
        print(f'{s}: {"10 batch reports inspected" if inspect_only else "10 batches mapped"}')
    if inspect_only:
        print('\nINSPECTION COMPLETE: no CSV files written. Review row identifiers above.')
        return
    data=pd.DataFrame(all_rows)
    assert len(data)==(191+188)*3
    # Aggregate only if exactly three valid replicate estimates per mutation/structure.
    counts=data.groupby(['structure','mutation']).agg(n=('ddg_kcal_mol','size'),n_runs=('replicate','nunique'))
    if not ((counts.n==3)&(counts.n_runs==3)).all(): raise ValueError('Expected exactly 3 distinct energy replicates for each mutation')
    if data.duplicated(['structure','mutation','replicate']).any():
        raise ValueError('Duplicate model identities after mapping')
    aggregate=(data.groupby(['structure','mutation','protein_position'],as_index=False)
              .agg(ddg_mean_kcal_mol=('ddg_kcal_mol','mean'),
                   ddg_median_kcal_mol=('ddg_kcal_mol','median'),
                   ddg_sd_kcal_mol=('ddg_kcal_mol','std'),
                   ddg_min_kcal_mol=('ddg_kcal_mol','min'),
                   ddg_max_kcal_mol=('ddg_kcal_mol','max'),
                   n_replicates=('ddg_kcal_mol','size')))
    cnt=aggregate.structure.value_counts().to_dict()
    if cnt!={'7SV7':191,'8EIG':188}: raise ValueError(f'Wrong structure mutation totals: {cnt}')
    target=(manifest[manifest.eligible==1][['pdb','mutation','protein_position']]
            .drop_duplicates().rename(columns={'pdb':'structure'}))
    check=aggregate.merge(target,on=['structure','mutation','protein_position'],how='outer',indicator=True)
    if (check._merge!='both').any(): raise ValueError('FoldX energies do not match Phase28E-1 mutation manifest')
    wide=aggregate.pivot(index='mutation',columns='structure',values='ddg_mean_kcal_mol')
    common=wide.dropna()
    if len(common)!=188: raise ValueError(f'Expected 188 mutations in both structures, got {len(common)}')
    print('\nA. STRUCTURAL COVERAGE')
    print('  7SV7:',cnt['7SV7'],'| 8EIG:',cnt['8EIG'],'| both:',len(common),'| 7SV7 only:',len(wide)-len(common))
    print('\nB. ΔΔG SUMMARY (kcal/mol): positive = destabilising in this FoldX template')
    print(aggregate.groupby('structure').ddg_mean_kcal_mol.describe().round(3).to_string())
    print('\nC. REPLICATE AGREEMENT / EXTREME VALUES')
    for s,g in aggregate.groupby('structure'):
        print(f'  {s}: median repeat SD={g.ddg_sd_kcal_mol.median():.3f}; '
              f'SD>2={sum(g.ddg_sd_kcal_mol>2)}; |ΔΔG|>20={sum(g.ddg_mean_kcal_mol.abs()>20)}')
    print('\nD. CROSS-STRUCTURE AGREEMENT (backgrounds differ)')
    print(f'  paired n={len(common)}; Pearson={common.corr(method="pearson").iloc[0,1]:.3f}; '
          f'Spearman={common.corr(method="spearman").iloc[0,1]:.3f}; '
          f'median abs difference={(common["7SV7"]-common["8EIG"]).abs().median():.3f} kcal/mol')
    delta=(common['7SV7']-common['8EIG']).abs().sort_values(ascending=False).head(12)
    print('  Largest disagreement:\n'+delta.round(3).to_string())
    print('\nE. LARGEST MAGNITUDE ΔΔG — QC candidates, NOT efficacy claims')
    print(aggregate.assign(abs_ddg=aggregate.ddg_mean_kcal_mol.abs()).sort_values('abs_ddg',ascending=False)
          [['structure','mutation','ddg_mean_kcal_mol','ddg_sd_kcal_mol']].head(14).round(3).to_string(index=False))
    # Write only after all batches and data schemas have validated.
    out.mkdir(parents=True,exist_ok=True)
    data.to_csv(out/'phase28e4_foldx_replicate_energies.csv',index=False)
    aggregate.to_csv(out/'phase28e4_foldx_mutation_energies.csv',index=False)
    wide.reset_index().to_csv(out/'phase28e4_foldx_structure_comparison.csv',index=False)
    pd.DataFrame(jobs).to_csv(out/'phase28e4_batch_provenance.csv',index=False)
    print('\nPASS: complete FoldX mutation-level energy extraction; no VTI outcomes accessed.')
    print('Saved:',out)


def self_test():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        w=Path(tmp); s='SAMPLE'
        names=[f'CFTR_Repair_{i}_{j}.pdb' for i in range(1,3) for j in range(3)]
        dif=w/'Dif_TEST_CFTR_Repair.fxout'; pdbl=w/'PdbList_TEST_CFTR_Repair.fxout'
        dif.write_text('FoldX 5 (2011)\nPdb\tTotal Energy\tBackbone Hbond\tSidechain Hbond\tVan der Waals\tElectrostatics\n'
                       +'\n'.join(f'{name}\t{0.5*k:.3f}\t1\t2\t3\t4' for k,name in enumerate(names))+'\n')
        pdbl.write_text('Pdb\tWT\n'+'\n'.join(f'{name}\tWT_{name}' for name in names)+'\n')
        rows,h=parse_dif(dif); assert len(rows)==6
        mapping=map_rows(rows,['AA10V;','GA11D;'],3,pdbl)
        assert len(mapping)==6 and mapping[-1]['foldx_mutation']=='GA11D;'
        # Reject wrong PdbList, since silent misalignment is unsafe.
        pdbl.write_text('Pdb\tWT\n'+'\n'.join(f'{name}\tWT_{name}' for name in names[::-1])+'\n')
        try:
            map_rows(rows,['AA10V;','GA11D;'],3,pdbl)
            raise AssertionError('Did not reject mismatched PdbList')
        except ValueError: pass
    print('PASS: FoldX numeric parsing, mutation/repeat assignment and mismatch rejection')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default='results/biology/phase28e')
    p.add_argument('--full',default='results/biology/phase28e/phase28e3_full')
    p.add_argument('--out',default='results/biology/phase28e/phase28e4')
    p.add_argument('--inspect',action='store_true',help='Inspect report formats only; no CSV written')
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test: self_test();return
    self_test()
    extract(Path(a.source),Path(a.full),Path(a.out),a.inspect)

if __name__=='__main__': main()
