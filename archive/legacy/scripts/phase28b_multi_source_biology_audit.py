#!/usr/bin/env python3
"""Phase 28B: inventory pre-treatment CFTR biology assays against frozen VTI TRAIN.

Read-only discovery. Uses VTI TRAIN mutation identities and TRAIN response *regimes*
for feasibility only. Does not read VTI VAL/TEST or fit any model. In particular,
NO treated assays, VTI labels, ETI labels, or numerical biology measurements are
exported as predictors. Column names are only provisional provenance clues.

Run from CFTR-Theratype root:
  python scripts/phase28b_multi_source_biology_audit.py --self-test
  python scripts/phase28b_multi_source_biology_audit.py --root data
  python scripts/phase28b_multi_source_biology_audit.py --root data \
      --extra-file /other/location/2024_supplement.xlsx

Output: results/biology/phase28b/phase28b_*.{csv,json}

Prerequisite: phase28a_trafficking_feature_audit.py already in scripts/.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / 'scripts'))
from phase28a_trafficking_feature_audit import parse_mutation, load_training

OUT = ROOT / 'results/biology/phase28b'
EXT = {'.xlsx','.xls','.csv','.tsv'}
# This is an exploratory inventory of OTHER sources, not VTI-outcome discovery.
BLOCKED_PATH_TERMS = ('/processed/','/results/','/bihler_2026','/vti','/vanzacaftor',
                      '/phase26','/phase27','/phase28','/test/')
AA_TOKEN = re.compile(r'^[ACDEFGHIKLMNPQRSTVWY]\d{1,4}[ACDEFGHIKLMNPQRSTVWY]$')
DRUG = ('vx-445','vx445','elexacaftor','elx','vx-661','vx661', 'tezacaftor',
        'tez','vx-809','vx809','lumacaftor','vx-121','vx121','vanzacaftor',
        'vti','eti','trikafta','alyftrek','treated','treatment','after drug',
        'corrector','rescue','increment','gain','fold change','delta response')
UNTREATED = ('dmso','untreated','vehicle','apo','basal','parental','control',
             'no drug','baseline','without drug')
FUNCTION = ('fsk','forskolin','chloride','current','function','transport')
BIOLOGY = ('band c','band_c','band b','band_b','c/(c+b)','c/b','surface','pme',
           'maturation','trafficking','expression','immunostain','calnexin','canx',
           'abundance','stability','protein')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def source_class(path):
    p=str(path).lower()
    if 'tedman' in p or '107180' in p:return 'Tedman / CANX'
    if 'jhangiani' in p or 'vx121' in p:return 'Jhangiani / VX121'
    if 'mcdonald' in p:return 'McDonald'
    if 'bihler' in p or 'mmc3' in p or 'mmc2' in p:return 'Bihler (verify year / condition)'
    return 'Unclassified — review publication'


def col_kind(col):
    s=re.sub(r'[_-]+',' ',str(col).lower())
    def has_kw(k):
        k=k.replace('_',' ').replace('-',' ')
        return re.search(r'(?<![a-z0-9])'+re.escape(k)+r'(?![a-z0-9])',s) is not None
    drug=any(has_kw(k) for k in DRUG)
    untreated=any(has_kw(k) for k in UNTREATED)
    bio=any(has_kw(k) for k in BIOLOGY)
    func=any(has_kw(k) for k in FUNCTION)
    if drug and not untreated:return 'DRUG-TREATED / RESPONSE — separate assay ladder'
    if drug and untreated:return 'MIXED/AMBIGUOUS treatment column — manual review'
    if bio and untreated:return 'CANDIDATE untreated biology — verify definition'
    if bio:return 'POTENTIAL biology — exposure not specified'
    if func and untreated:return 'UNTREATED function — separate assay-informed setting'
    if func:return 'FUNCTION/ASSAY — exposure unknown'
    if untreated:return 'UNTREATED / CONTROL — inspect assay meaning'
    return 'OTHER / UNCLASSIFIED'


def iter_tables(path, max_sheets, max_rows):
    if path.suffix.lower() in ('.xlsx','.xls'):
        book=pd.ExcelFile(path)
        for sheet in book.sheet_names[:max_sheets]:
            for header in range(5):
                try:
                    df=pd.read_excel(book,sheet_name=sheet,header=header,nrows=max_rows)
                except Exception:continue
                if not df.empty:
                    yield str(sheet),header,df
    else:
        sep='\t' if path.suffix.lower()=='.tsv' else ','
        for hdr in range(3):
            try:
                df=pd.read_csv(path,sep=sep,header=hdr,nrows=max_rows,low_memory=False,
                               encoding_errors='replace')
            except Exception:continue
            if not df.empty: yield 'flat_table',hdr,df


def choose_sheet_candidate(path, train_codes, *,max_sheets=12,max_rows=5000):
    best=None
    for sheet,header,df in iter_tables(path,max_sheets,max_rows):
        if df.shape[1]>220: df=df.iloc[:,:220]  # protective cap
        for col in df.columns:
            series=df[col]
            if series.notna().sum()<4:continue
            # Restrict to text-like mutation identifier columns; avoid iterating
            # numerical measurement arrays unnecessarily.
            if pd.api.types.is_numeric_dtype(series):continue
            names=series.map(parse_mutation)
            valid=names.dropna()
            if len(valid)<4:continue
            matched=set(valid)&train_codes
            # Prefer most TRAIN overlap, then most parseable mutations.
            score=(len(matched),len(valid),-header)
            if best is None or score>best['score']:
                byname={v for v in valid if AA_TOKEN.fullmatch(str(v))}
                best={'score':score,'file':str(path),'sheet':sheet,'header':int(header),
                      'mutation_col':str(col),'source_n':int(len(df)),
                      'parseable_n':int(len(valid)), 'source_unique_n':len(byname),
                      'matching_codes':matched,'measurement_columns':[
                          {'column':str(c),'classification':col_kind(c)}
                          for c in df.columns if c!=col],
                      'source':source_class(path)}
    return best


def discover(root, extras, max_files, max_mb):
    candidates=[]
    if root.is_dir():candidates.extend(p for p in root.rglob('*') if p.is_file())
    for p in extras:
        if p.is_file():candidates.append(p)
        else:print(f'WARNING: extra file not found: {p}')
    seen=set(); out=[]; skipped=[]
    for p in sorted(candidates):
        if p.suffix.lower() not in EXT or p.name.startswith('~$'):continue
        rp=p.resolve()
        if rp in seen:continue
        seen.add(rp)
        low='/' + str(p).lower().replace('\\','/')
        if any(s in low for s in BLOCKED_PATH_TERMS):
            skipped.append((str(p),'blocked possible VTI/protected source'));continue
        if p.stat().st_size > max_mb*1024*1024:
            skipped.append((str(p),f'too large: >{max_mb} MB'));continue
        out.append(p)
    if len(out)>max_files:
        skipped.extend((str(p),'max-files cap') for p in out[max_files:]);out=out[:max_files]
    return out,skipped


def summarize_one(candidate,train):
    codes=candidate['matching_codes']
    t=train[train.mutation.isin(codes)]
    cnt=t.regime.value_counts()
    out={k:v for k,v in candidate.items() if k not in ('score','matching_codes','measurement_columns')}
    out.update({'matched_unique':len(codes),'matched_records':len(t),
                'matched_positions':int(t.protein_position.nunique()),
                'low':int(cnt.get('low',0)), 'middle':int(cnt.get('middle',0)),
                'high':int(cnt.get('high',0)),
                'possible_untreated_cols':sum(z['classification'].startswith('CANDIDATE untreated')
                                              for z in candidate['measurement_columns']),
                'other_biology_cols':sum('biology' in z['classification'].lower()
                                          for z in candidate['measurement_columns'])})
    return out


def audit(root, extras, max_files, max_mb, max_sheets, max_rows):
    OUT.mkdir(parents=True,exist_ok=True)
    train=load_training()
    if not len(train)==210 or train.mutation.nunique()!=206:
        raise ValueError('Unexpected frozen TRAIN identity')
    candidates,skipped=discover(root,extras,max_files,max_mb)
    print('='*115)
    print('PHASE 28B | MULTI-SOURCE CFTR BIOLOGY FEASIBILITY | TRAIN ONLY | NO MODEL FIT')
    print('='*115)
    print('VTI TRAIN 210 records / 206 protein substitutions / 168 positions; regimes: 52/129/29')
    print('Files selected:',len(candidates),'blocked/skipped:',len(skipped))
    print('No VTI VAL/TEST read. No treatment-response column values or numeric assay values exported.')
    print('Only exact protein substitution strings are joined; no position-only matching.')
    print('Column names alone CANNOT establish untreated status, units, or biological comparability.')
    records=[];details=[];merged=set()
    for i,p in enumerate(candidates,1):
        try:candidate=choose_sheet_candidate(p,set(train.mutation),max_sheets=max_sheets,max_rows=max_rows)
        except Exception as ex:
            skipped.append((str(p),f'{type(ex).__name__}: {str(ex)[:120]}'));continue
        if candidate is None:continue
        s=summarize_one(candidate,train)
        s.update({'source_sha256':digest(p)})
        records.append(s);merged|=candidate['matching_codes']
        for obj in candidate['measurement_columns']:
            details.append({'file':str(p),'sheet':candidate['sheet'],
                            'source':candidate['source'],**obj})
        if s['matched_records'] or s['possible_untreated_cols']:
            print(f'[{i}/{len(candidates)}] {str(p)[-77:]} | sheet={s["sheet"]} '
                  f'key={s["mutation_col"]!r} parsed={s["source_unique_n"]} '
                  f'matched={s["matched_records"]} pos={s["matched_positions"]} '
                  f'L/M/H={s["low"]}/{s["middle"]}/{s["high"]} '
                  f'candidate_untreated_cols={s["possible_untreated_cols"]}')
            shown=0
            for z in candidate['measurement_columns']:
                if z['classification'].startswith('CANDIDATE untreated') and shown<8:
                    print('   ',z['column'][:85]);shown+=1
    summary=pd.DataFrame(records)
    if len(summary):
        summary=summary.sort_values(['matched_records','high','possible_untreated_cols'],ascending=False)
        summary.to_csv(OUT/'phase28b_source_coverage.csv',index=False)
    else:pd.DataFrame(columns=['file','matched_records']).to_csv(OUT/'phase28b_source_coverage.csv',index=False)
    pd.DataFrame(details,columns=['file','sheet','source','column','classification']).to_csv(
        OUT/'phase28b_measurement_provenance_to_review.csv',index=False)
    pd.DataFrame(skipped,columns=['file','reason']).to_csv(OUT/'phase28b_skipped_files.csv',index=False)
    cover=train[['variant_key','mutation','protein_position','regime']].copy()
    cover['has_any_candidate_source']=cover.mutation.isin(merged)
    cover.to_csv(OUT/'phase28b_train_overlap_flags.csv',index=False)
    c=cover[cover.has_any_candidate_source]
    ct=c.regime.value_counts()
    print('\nA. TOP SOURCE OVERLAP (not yet vetted for biological assay eligibility)')
    if len(summary):
        cols=['source','file','sheet','matched_records','matched_positions','low','middle','high','possible_untreated_cols']
        with pd.option_context('display.max_colwidth',65,'display.width',220):
            print(summary[cols].head(20).to_string(index=False))
    else: print('No parseable source files. Supply --extra-file on a known downloaded supplement.')
    print('\nB. UNION COVERAGE — optimistic upper bound across raw source files')
    print(f'{len(c)}/210 records, {c.mutation.nunique()}/206 substitutions, '
          f'{c.protein_position.nunique()}/168 positions; L/M/H='
          f'{int(ct.get("low",0))}/{int(ct.get("middle",0))}/{int(ct.get("high",0))}')
    print('CAUTION: This union includes treated and unvetted studies; it is NOT eligible-feature coverage.')
    print('\nC. DECISION')
    elig=summary[(summary.matched_records>=30)&(summary.possible_untreated_cols>=1)] if len(summary) else pd.DataFrame()
    if len(elig):
        print('Potential sources with >=30 matched records AND a candidate untreated biology column:')
        print(elig[['file','matched_records','low','middle','high','possible_untreated_cols']].to_string(index=False))
        print('NEXT: manually verify assay meaning, condition, units, source, uniqueness before modelling.')
    else:
        print('No source found with >=30 matches and an identifiable untreated biology column.')
        print('NEXT: obtain a larger matching baseline-maturation dataset OR focus on mutation-derived biophysics.')
    print('\nDo not use treated Band C, VX445/661 response, or any VTI functional outcome as a mutation-only input.')
    manifest={'status':'discovery_only','protected_labels_read':False,
              'models_fit':False,'root':str(root),'files_examined':len(candidates),
              'source_files_matched':int(len(records)),
              'union_records':int(len(c)),'union_high':int(ct.get('high',0)),
              'files':[{k:v for k,v in r.items() if k!='source_sha256'} for r in records]}
    (OUT/'phase28b_manifest.json').write_text(json.dumps(manifest,indent=2,default=str))
    print('PASS Phase 28B discovery. Saved:',OUT)


def self_test():
    assert col_kind('Untreated Band C abundance').startswith('CANDIDATE untreated')
    assert col_kind('DMSO surface expression').startswith('CANDIDATE untreated')
    assert col_kind('VX-445 treated Band C').startswith('DRUG-TREATED')
    assert col_kind('VX-661 DMSO surface').startswith('MIXED')
    assert parse_mutation('p.Val520Phe')=='V520F'
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as d:
        p=Path(d)/'mcdonald_example.xlsx'
        pd.DataFrame({'Variant':['V520F','G85E','R560T','P67L','Y1032C','P67L'],
                      'Untreated Band C abundance':[0.2,0.1,0.3,0.4,0.5,0.4],
                      'VX-445 response':[1.,2.,3.,4.,5.,4.]}).to_excel(p,index=False)
        got=choose_sheet_candidate(p,{'V520F','G85E','Y1032C'})
        assert got and len(got['matching_codes'])==3
        tr=pd.DataFrame({'mutation':['V520F','G85E','Y1032C'],'protein_position':[520,85,1032],
                         'regime':['low','middle','high']})
        s=summarize_one(got,tr)
        assert (s['matched_records'],s['low'],s['high'],s['possible_untreated_cols'])==(3,1,1,1),s
    print('PASS synthetic: mutation matching, treatment-aware header warnings, overlap, group accounting')


if __name__=='__main__':
    pa=argparse.ArgumentParser()
    pa.add_argument('--root',type=Path,default=Path('data'))
    pa.add_argument('--extra-file',type=Path,action='append',default=[])
    pa.add_argument('--max-files',type=int,default=100)
    pa.add_argument('--max-size-mb',type=int,default=30)
    pa.add_argument('--max-sheets',type=int,default=12)
    pa.add_argument('--max-rows',type=int,default=5000)
    pa.add_argument('--self-test',action='store_true')
    args=pa.parse_args()
    if args.self_test:self_test()
    else:audit(args.root,args.extra_file,args.max_files,args.max_size_mb,
               args.max_sheets,args.max_rows)
