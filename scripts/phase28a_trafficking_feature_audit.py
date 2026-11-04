#!/usr/bin/env python3
"""Phase 28A: audit independent CFTR trafficking / proteostasis features.

Source: Tedman et al., eLife 2025, Supplementary file 2.
https://elifesciences.org/articles/107180v1/figures
https://cdn.elifesciences.org/articles/107180/elife-107180-supp2-v1.xlsx

Purpose: discover exact missense overlap with VTI TRAIN, sheet schemas, and
experimental-measurement provenance BEFORE fitting any models.

SECURITY: reads VTI TRAIN via Phase26H loader; never reads VAL/TEST outcomes.
DO NOT use VTI-treated maturation or any outcome-derived response as a
mutation-only feature. A measured DMSO/parental expression assay is an
'additional assay known' scenario, NOT a sequence-only prediction.

Usage:
  python scripts/phase28a_trafficking_feature_audit.py --self-test
  python scripts/phase28a_trafficking_feature_audit.py \\
    --xlsx data/external/tedman2025_canx/elife-107180-supp2-v1.xlsx
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
OUT = ROOT / 'results/biology/phase28a'
DEFAULT_XLSX = ROOT / 'data/external/tedman2025_canx/elife-107180-supp2-v1.xlsx'
URL = 'https://cdn.elifesciences.org/articles/107180/elife-107180-supp2-v1.xlsx'
AA3 = {
    'Ala':'A','Cys':'C','Asp':'D','Glu':'E','Phe':'F','Gly':'G','His':'H',
    'Ile':'I','Lys':'K','Leu':'L','Met':'M','Asn':'N','Pro':'P','Gln':'Q',
    'Arg':'R','Ser':'S','Thr':'T','Val':'V','Trp':'W','Tyr':'Y',
}
VALID_AA = set('ACDEFGHIKLMNPQRSTVWY')

# Warning terms used only to label measurement columns for manual review.
# We intentionally DO NOT auto-select numeric values for modelling.
EXPOSURE_TERMS = [
    '445', '661', '809', '121', 'modulator', 'corrector',
    'drug', 'treated', 'treatment', 'fold change', 'response', 'rescue',
    'delta', 'gain', 'effect', 'effect size', 'ratio',
]
BASELINE_TERMS = ['dmso', 'vehicle', 'untreated', 'basal', 'control', 'parental']
CHAPERONE_TERMS = ['canx', 'calnexin', 'knockout', 'ko', 'kd', 'knockdown']


def parse_mutation(text):
    """Conservative exact single-AA substitution parser (no site-only joins)."""
    if pd.isna(text):
        return None
    t = str(text).strip()
    if not t or t.lower() in ('nan','none','wild type','wild-type','wt'):
        return None
    t = t.replace('Δ', 'del').replace('−', '-')
    one = list(re.finditer(r'(?<![A-Za-z0-9])([A-Z])(\d{1,4})([A-Z])(?![A-Za-z0-9])', t))
    one = [m for m in one if m.group(1) in VALID_AA and m.group(3) in VALID_AA]
    # Accept: V520F; p.V520F; CFTR p.V520F; not multivariant genotypes.
    if len(one) == 1:
        m = one[0]
        if 1 <= int(m.group(2)) <= 1480 and not re.search(r'[/;+]', t):
            return m.group(1) + str(int(m.group(2))) + m.group(3)
    # Three-letter HGVS notation.
    three = re.findall(r'([A-Z][a-z]{2})(\d{1,4})([A-Z][a-z]{2})', t)
    if len(three) == 1 and not re.search(r'[/;+]', t):
        a, p, b = three[0]
        if a in AA3 and b in AA3 and 1 <= int(p) <= 1480:
            return AA3[a] + str(int(p)) + AA3[b]
    return None


def sha256(p):
    h = hashlib.sha256()
    with p.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def measurement_kind(name):
    s = str(name).lower().replace('_',' ').replace('-',' ')
    if any(term in s for term in EXPOSURE_TERMS):
        return 'TREATED / DERIVED (do NOT auto-use as feature)'
    if any(re.search(r'(?<!\w)' + re.escape(term) + r'(?!\w)', s) for term in CHAPERONE_TERMS):
        return 'CANX-PERTURBATION (extra assay scenario)'
    if any(term in s for term in BASELINE_TERMS):
        return 'POSSIBLE BASELINE ASSAY (review units/conditions)'
    return 'UNKNOWN (requires manual provenance review)'


def load_training():
    from cftr_theratype.datasets.eti_auxiliary import load_data as load_eti_train
    from cftr_theratype.features.mutation_codes import extract_codes
    v, x, _, _, folds = load_eti_train(ROOT)
    if len(v) != 210 or v.protein_position.nunique() != 168 or len(folds) != 10:
        raise ValueError('TRAIN partition changed; expected 210 / 168 / 10 folds')
    pp = v.protein_position.to_numpy(int)
    codes = extract_codes(v, pp)
    t = pd.DataFrame({'variant_key': v.variant_key.astype(str).to_numpy(),
                      'protein_position': pp, 'mutation': codes,
                      'corrector_rescue_pp': v.corrector_rescue_pp.to_numpy(float)})
    t['regime'] = np.select([t.corrector_rescue_pp < 10, t.corrector_rescue_pp > 40],
                             ['low', 'high'], default='middle')
    if t.variant_key.duplicated().any() or t.mutation.nunique() != 206:
        raise ValueError('TRAIN mutation identity unexpectedly changed')
    if not np.isfinite(t.corrector_rescue_pp).all():
        raise ValueError('TRAIN corrector labels not finite')
    return t


def evaluate_sheet(xls, sheet, train_codes):
    candidates = []
    for hdr in range(6):
        try:
            df = pd.read_excel(xls, sheet_name=sheet, header=hdr)
        except Exception as exc:
            print(f'  NOTE header {hdr}: {type(exc).__name__}: {str(exc)[:120]}')
            continue
        if df.empty:
            continue
        for col in df.columns:
            # The first ~1000 rows suffice to locate a key candidate; exact
            # coverage is recomputed for the selected column over all rows.
            sampled = df[col].dropna().head(1000)
            if len(sampled) < 4:
                continue
            codes = sampled.map(parse_mutation)
            count = int(codes.notna().sum())
            hits = len(set(codes.dropna()) & train_codes)
            if count >= 4:
                candidates.append((hits, count, hdr, str(col), df))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1], -x[2]), reverse=True)
    hits, count, hdr, key_col, df = candidates[0]
    normalized = df[key_col].map(parse_mutation)
    assigned = df.copy()
    assigned.insert(0, '__cftr_mutation', normalized)
    valid = assigned.loc[assigned.__cftr_mutation.notna()].copy()
    overlap = valid.__cftr_mutation.isin(train_codes)
    return {
        'sheet': sheet, 'header_row_zero_based': hdr, 'variant_col': key_col,
        'rows_total': int(len(df)), 'parseable_rows': int(len(valid)),
        'unique_protein_substitutions': int(valid.__cftr_mutation.nunique()),
        'overlap_unique': int(valid.loc[overlap, '__cftr_mutation'].nunique()),
        'overlap_rows': int(overlap.sum()),
        'duplicated_mutation_rows': int(valid.__cftr_mutation.duplicated().sum()),
        'columns': list(map(str, df.columns)),
        'data': assigned,
        'candidate_columns': [(str(c), measurement_kind(c)) for c in df.columns if str(c) != key_col],
    }


def audit(path: Path, top_sheets: int=20):
    if not path.is_file():
        raise FileNotFoundError(
            f'Cannot locate {path}\n'
            f'Download the published Supplementary file 2 from:\n{URL}\n'
            'Then rerun --xlsx PATH. Do not substitute Supplementary file 1 (read counts).'
        )
    OUT.mkdir(parents=True, exist_ok=True)
    tr = load_training()
    tcodes = set(tr.mutation)
    xls = pd.ExcelFile(path)
    print('='*112)
    print('PHASE 28A | EXPERIMENTAL TRAFFICKING / CALNEXIN FEATURE FEASIBILITY AUDIT')
    print('='*112)
    print('Input eLife Supplementary 2:', path)
    print('SHA256:', sha256(path))
    print('VTI TRAIN: 210 records / 206 substitutions / 168 protein positions')
    print('Original 2x5 position-disjoint folds retained; NO VAL or TEST read.')
    print('NO predictive model fit. No assay-response data used as model inputs.')
    print('Sheets:', xls.sheet_names)
    all_results, rows, best_codes = [], [], set()
    for name in xls.sheet_names[:top_sheets]:
        result = evaluate_sheet(xls, name, tcodes)
        if result is None:
            print(f'\nSHEET [{name}]: no convincing single-substitution identifier in first six header rows')
            rows.append({'sheet': name, 'parsed':0, 'overlap_unique':0, 'header':None, 'mutation_key':None})
            continue
        data = result.pop('data')
        print(f'\nSHEET [{name}] header={result["header_row_zero_based"]}, '
              f'key={result["variant_col"]!r} | rows={result["rows_total"]} '
              f'parseable={result["parseable_rows"]}, unique={result["unique_protein_substitutions"]}')
        print('  Overlap: unique substitutions=', result['overlap_unique'],
              ' source rows=', result['overlap_rows'],
              ' duplicate substitution rows=', result['duplicated_mutation_rows'])
        print('  Source measurement fields / provenance review:')
        for col, category in result['candidate_columns'][:55]:
            print(f'    {col[:72]:72s} {category}')
        if len(result['candidate_columns']) > 55:
            print(f'    ... {len(result["candidate_columns"])-55} additional columns')
        matched = set(data.__cftr_mutation.dropna()) & tcodes
        if result['overlap_unique'] >= 1:
            q = tr[tr.mutation.isin(matched)].copy()
            n_by = q.regime.value_counts().reindex(['low','middle','high'],fill_value=0)
            print('  Matched VTI TRAIN records:',len(q),' / positions:',q.protein_position.nunique(),
                  '/ regimes (low,mid,high):',list(n_by.to_numpy(int)))
            rows.append({'sheet':name, 'parsed':result['parseable_rows'],
                         'overlap_unique':result['overlap_unique'],
                         'overlap_train_records':len(q),
                         'overlap_positions':q.protein_position.nunique(),
                         'low':int(n_by['low']), 'middle':int(n_by['middle']), 'high':int(n_by['high']),
                         'header':result['header_row_zero_based'], 'mutation_key':result['variant_col']})
        else:
            rows.append({'sheet':name, 'parsed':result['parseable_rows'], 'overlap_unique':0,
                         'header':result['header_row_zero_based'], 'mutation_key':result['variant_col']})
        all_results.append(result)
        best_codes.update(matched)
    # Mutation-level source coverage only; no numerical external assay values
    # are exported until their provenance/units are confirmed.
    tr['has_tedman_2025_source'] = tr.mutation.isin(best_codes)
    tr[['variant_key','mutation','protein_position','regime','has_tedman_2025_source']].to_csv(
        OUT / 'phase28a_train_source_coverage.csv',index=False)
    pd.DataFrame(rows).to_csv(OUT / 'phase28a_sheet_overlap.csv',index=False)
    clean = [{k: v for k,v in z.items() if k not in ('columns','candidate_columns')} for z in all_results]
    manifest = {'source_url': URL, 'source_file':str(path),'source_sha256':sha256(path),
                'train_coverage_unique':int(len(best_codes)),
                'train_coverage_records':int(tr.has_tedman_2025_source.sum()),
                'protected_labels_read':False, 'predictive_model_fit':False,
                'sheets':clean}
    (OUT / 'phase28a_source_manifest.json').write_text(json.dumps(manifest,indent=2))
    coverage = tr.groupby('regime', sort=False)['has_tedman_2025_source'].agg(['sum','count'])
    print('\n'+'='*112)
    print('FINAL UNIQUE MUTATION-LEVEL SOURCE COVERAGE (union across sheets)')
    print('='*112)
    print(f'Matched TRAIN records: {int(tr.has_tedman_2025_source.sum())}/210; '
          f'unique substitutions: {len(best_codes)}/206; '
          f'unique protein positions: {tr.loc[tr.has_tedman_2025_source,"protein_position"].nunique()}/168')
    print(coverage.to_string())
    print('\nNEXT: inspect exact sheet/column provenance before proposing a feature. Do not infer the input from column names alone.')
    print('Recommended priorities: parent-cell DMSO untreated surface expression; CANX KO dependence in DMSO;')
    print('then ETI VX-445/661 treated PME in a separate assay-informed ladder (not mutation-only).')
    print('CRITICAL: no VTI-treated band C or VTI functional outcomes as predictors for VTI prediction.')
    print('Saved:', OUT)
    print('PASS Phase 28A source audit; no protected labels opened or predictive model trained.')


def self_test():
    x = ['V520F','p.V520F','p.Val520Phe','CFTR p.Gly85Glu',
         'F508del', 'V520F/R560T', 'WT', 'Y1032C']
    got = [parse_mutation(i) for i in x]
    assert got == ['V520F','V520F','V520F','G85E',None,None,None,'Y1032C'],got
    assert measurement_kind('VX-445 response').startswith('TREATED')
    assert measurement_kind('CANX KO expression').startswith('CANX')
    assert measurement_kind('Parental DMSO').startswith('POSSIBLE')
    # Synthetic two-sheet xlsx tests header detection and exact missense overlap.
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmp:
        p = Path(tmp) / 'test.xlsx'
        with pd.ExcelWriter(p) as writer:
            pd.DataFrame({'CFTR mutation':['V520F','p.Val520Phe','G85E','R560T','F508del','P67L'],
                          'DMSO parental':[1,2,3,4,5,6],
                          'VX445 treated':[2,3,4,5,6,7]}).to_excel(writer, sheet_name='Assays',index=False)
        xx = pd.ExcelFile(p)
        res = evaluate_sheet(xx,'Assays', {'V520F','G85E','I123T'})
        assert res['overlap_unique']==2 and res['overlap_rows']==3, res
        assert res['header_row_zero_based']==0
    print('PASS synthetic: HGVS substitution keys, exclusions, source columns, overlapping sheet scan')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--xlsx',type=Path,default=DEFAULT_XLSX)
    ap.add_argument('--top-sheets',type=int,default=20)
    ap.add_argument('--self-test',action='store_true')
    a=ap.parse_args()
    if a.self_test:
        self_test()
    else:
        audit(a.xlsx,a.top_sheets)
