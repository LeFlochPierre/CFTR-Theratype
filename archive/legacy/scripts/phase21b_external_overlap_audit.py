#!/usr/bin/env python3
"""Phase 21B: external Tedman 2025 CFTR surface-expression audit.

NO VTI/ETI outcomes are opened. This creates candidate external auxiliary labels,
not a CV training table: each outer fold must additionally exclude its held-out
protein positions. It deliberately does not merge external intensity measurements
into protected partition records.
"""
from pathlib import Path
import hashlib
import json
import re
import numpy as np
import pandas as pd

ROOT = Path.home() / 'projects/CFTR-Theratype'
BOOK = ROOT / 'data/external/tedman2025/cftr_expression.xlsx'
SPLITS = ROOT / 'data/processed/bihler_2026_transfer_splits.parquet'
OUT = ROOT / 'results/biology/phase21b'
OUT.mkdir(parents=True, exist_ok=True)

AA3 = 'Ala Arg Asn Asp Cys Glu Gln Gly His Ile Leu Lys Met Phe Pro Ser Thr Trp Tyr Val'.split()
AA1 = 'A R N D C E Q G H I L K M F P S T W Y V'.split()
TO1 = dict(zip(AA3, AA1))
ONE = set(AA1)


def protein_id(value):
    """Return (WT, 1-based position, alternate, canonical short form) or none."""
    if pd.isna(value):
        return None
    s = re.sub(r'\s+', '', str(value)).removeprefix('p.')
    m = re.fullmatch(r'([A-Z])(\d+)([A-Z])', s)
    if m and m.group(1) in ONE and m.group(3) in ONE:
        wt, pos, alt = m.group(1), int(m.group(2)), m.group(3)
        return wt, pos, alt, f'{wt}{pos}{alt}'
    m = re.fullmatch(r'([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})', s)
    if m and m.group(1) in TO1 and m.group(3) in TO1:
        wt, pos, alt = TO1[m.group(1)], int(m.group(2)), TO1[m.group(3)]
        return wt, pos, alt, f'{wt}{pos}{alt}'
    return None


RE_VALUE = re.compile(
    r'^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))'
    r'(?:\s*(?:±|\+/-)\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)))?\s*$'
)


def parse_expression(v):
    if pd.isna(v) or not str(v).strip():
        return np.nan, np.nan, 'missing'
    if isinstance(v, (float, int, np.number)) and np.isfinite(v):
        return float(v), np.nan, 'mean_only'
    s = str(v).replace(',', '').replace('−', '-').strip()
    m = RE_VALUE.fullmatch(s)
    if m:
        return float(m.group(1)), (float(m.group(2)) if m.group(2) is not None else np.nan), 'parsed'
    return np.nan, np.nan, 'unparsed'


def as_bool(series):
    return series.fillna(False).map(lambda x: str(x).lower() in ('1','true','t','yes'))


assert BOOK.is_file(), f'Missing {BOOK}'
assert SPLITS.is_file(), f'Missing {SPLITS}'
raw = pd.read_excel(BOOK, sheet_name='Sheet4')
mem = pd.read_parquet(SPLITS)
assert {'Variant', 'CFTR sub-domain'}.issubset(raw.columns)
assert {'variant_key', 'hgvs_protein', 'protein_position', 'transfer_split'}.issubset(mem.columns)
assert mem.variant_key.is_unique, 'Frozen membership identity not unique'

print('=' * 101)
print('PHASE 21B | TEDMAN 2025 CFTR EXTERNAL BIOLOGY & POSITION OVERLAP AUDIT')
print('=' * 101)
print('Workbook SHA256:', hashlib.sha256(BOOK.read_bytes()).hexdigest())
print('Workbook rows:', len(raw))
print('Frozen VTI membership rows:', len(mem))
print('No VTI or ETI experimental outcomes accessed')

raw = raw.copy()
raw['external_variant'] = raw['Variant'].astype('string')
parsed = raw['Variant'].map(protein_id)
raw['variant_short'] = parsed.map(lambda x: x[3] if x else None)
raw['wt_aa'] = parsed.map(lambda x: x[0] if x else None)
raw['protein_position'] = parsed.map(lambda x: x[1] if x else pd.NA).astype('Int64')
raw['mut_aa'] = parsed.map(lambda x: x[2] if x else None)

mem = mem.copy()
mem['variant_short'] = mem.hgvs_protein.map(lambda s: (protein_id(s) or [None] * 4)[3])
if mem.variant_short.isna().any():
    print('WARNING: nonstandard membership protein HGVS:', int(mem.variant_short.isna().sum()))

# Check external WT amino acids against the actual project's CFTR assay construct.
# This does not attempt to silently remap M470/V470 or other discrepancies.
from cftr_theratype.reference import bihler_2024_assay_sequence
seq = bihler_2024_assay_sequence()
raw['reference_aa_match'] = raw.apply(
    lambda r: (str(seq[int(r.protein_position)-1]) == r.wt_aa)
    if pd.notna(r.protein_position) and 1 <= int(r.protein_position) <= len(seq)
    else pd.NA, axis=1
)

conditions = ['Apo', 'VX-661', 'VX-445', 'VX-445 + VX-661']
conditions_ids = ['apo', 'vx661', 'vx445', 'vx445_vx661']
status_counts = {}
nonparsed_examples = []
for prefix, short in [('Parental', 'parent'), ('CANX KO', 'canx_ko')]:
    for cond, cond_id in zip(conditions, conditions_ids):
        col = f'{prefix} Surface Immuno-staining Intensity ({cond})'
        assert col in raw, f'Expected workbook column missing: {col}'
        triple = raw[col].map(parse_expression)
        stem = f'{short}_{cond_id}'
        raw[stem + '_mean'] = triple.map(lambda v: v[0])
        raw[stem + '_pm'] = triple.map(lambda v: v[1])
        statuses = triple.map(lambda v: v[2])
        status_counts[stem] = statuses.value_counts(dropna=False).to_dict()
        bad = raw.loc[statuses.eq('unparsed'), ['external_variant', col]].head(4)
        for rec in bad.to_dict('records'):
            nonparsed_examples.append(dict(field=stem, **rec))

train_pos = set(mem.loc[mem.transfer_split.eq('train'), 'protein_position'].dropna().astype(int))
val_pos = set(mem.loc[mem.transfer_split.eq('val'), 'protein_position'].dropna().astype(int))
test_pos = set(mem.loc[mem.transfer_split.eq('test'), 'protein_position'].dropna().astype(int))
excluded_pos = set(mem.loc[mem.transfer_split.eq('excluded'), 'protein_position'].dropna().astype(int))
protected_2024 = set()
if 'protected_2024_validation_or_test_position' in mem:
    protected_2024 = set(mem.loc[as_bool(mem.protected_2024_validation_or_test_position), 'protein_position'].dropna().astype(int))
nontrain = val_pos | test_pos | excluded_pos | protected_2024

# Positions are unique across model folds, but external variants may overlap any VTI partition.
variant_to_partition = mem.dropna(subset=['variant_short']).groupby('variant_short')['transfer_split'].agg(
    lambda x: ';'.join(sorted(set(x.astype(str))))
).to_dict()
raw['same_substitution_in_vti'] = raw.variant_short.map(variant_to_partition).fillna('no')
raw['vti_train_position'] = raw.protein_position.isin(train_pos)
raw['vti_val_position'] = raw.protein_position.isin(val_pos)
raw['vti_test_position'] = raw.protein_position.isin(test_pos)
raw['vti_excluded_position'] = raw.protein_position.isin(excluded_pos)
raw['protected_2024_position'] = raw.protein_position.isin(protected_2024)
raw['safe_initial_aux_candidate'] = (
    raw.protein_position.notna()
    & raw.reference_aa_match.eq(True)
    & ~raw.protein_position.isin(nontrain)
)

assay_mean_cols = [f'{short}_{cid}_mean'
    for short in ['parent','canx_ko'] for cid in conditions_ids]
assay_pm_cols = [c.replace('_mean','_pm') for c in assay_mean_cols]

print('\nVARIANT IDENTITY AND OVERLAP')
print('External rows:',len(raw))
print('Parseable single-AA substitutions:',int(raw.variant_short.notna().sum()))
print('Unique parseable substitutions:',raw.variant_short.nunique())
print('External substitutions repeated:', int(raw.variant_short.dropna().duplicated().sum()))
print('External WT-AA matches project reference:',int(raw.reference_aa_match.eq(True).sum()))
print('External WT-AA mismatches:',int(raw.reference_aa_match.eq(False).sum()))
print('Distinct exact matches to a VTI-listed protein substitution:',int(raw.loc[raw.same_substitution_in_vti.ne('no'),'variant_short'].nunique()))
for part, pos in [('TRAIN',train_pos),('VAL',val_pos),('TEST',test_pos),('EXCLUDED',excluded_pos),('2024 protected',protected_2024)]:
    subset=raw[raw.protein_position.isin(pos)]
    print(f'{part:16s} external rows={len(subset):3d} positions={subset.protein_position.nunique():3d} '
          f'exact substitution matches={int(subset.same_substitution_in_vti.ne("no").sum()):3d}')
new=raw.loc[raw.protein_position.notna() & ~raw.protein_position.isin(train_pos | nontrain)]
print('External variants at new/unprotected positions:',len(new),f'({new.protein_position.nunique()} positions)')

print('\nEXTERNAL ASSAY COVERAGE (mean / ± as reported)')
for stem in assay_mean_cols:
    print(f'{stem:28s} mean={raw[stem].notna().sum():3d}/{len(raw)}, '
          f'±={raw[stem.replace("_mean","_pm")].notna().sum():3d}/{len(raw)}')

print('\nUNPARSED / QC ISSUES')
print('Unparsed external substitutions:',raw.loc[raw.variant_short.isna(),'Variant'].astype(str).head(20).tolist())
print('Reference-AA mismatches:',raw.loc[raw.reference_aa_match.eq(False),['Variant','protein_position']].head(20).to_dict('records'))
print('Unparsed measurement examples:',nonparsed_examples[:12])
print('Rows with all 8 external means:',int(raw[assay_mean_cols].notna().all(axis=1).sum()))

# Avoid merging exact nucleotide records with external protein variants here.
# A variant may have multiple cDNA identities; that must be addressed downstream.
metadata_cols=['external_variant','variant_short','protein_position','wt_aa','mut_aa',
               'CFTR sub-domain','reference_aa_match','same_substitution_in_vti',
               'vti_train_position','vti_val_position','vti_test_position',
               'vti_excluded_position','protected_2024_position','safe_initial_aux_candidate']
cols = metadata_cols + assay_mean_cols + assay_pm_cols
allout=OUT/'tedman2025_membership_audit.csv'
safeout=OUT/'tedman2025_initial_aux_candidates.csv'
# Membership audit is METADATA-ONLY. Do not emit external measurements for
# protected VTI or ETI positions into development results.
raw[metadata_cols].to_csv(allout,index=False)
# This candidate pool excludes ALL protected/nontraining positions globally.
# MUST ALSO exclude each outer fold's held-out VTI positions during multitask training.
safe=raw.loc[raw.safe_initial_aux_candidate].copy()
safe[cols].to_csv(safeout,index=False)
print('\nSAFE CANDIDATE POOL (NOT YET FOLD-FILTERED)')
print('Auxiliary external variants:',len(safe),'distinct positions:',safe.protein_position.nunique())
print('At VTI TRAIN positions:',int(safe.vti_train_position.sum()))
print('At novel unprotected positions:',int((~safe.vti_train_position).sum()))
print('Potential data overlaps val/test/excluded/2024-protected positions: 0 by construction')
print('IMPORTANT: Drop the outer held-out VTI protein positions from this pool IN EACH FOLD.')
print('Saved:',allout)
print('Saved:',safeout)
meta=OUT/'phase21b_audit_summary.json'
meta.write_text(json.dumps(dict(source_sha256=hashlib.sha256(BOOK.read_bytes()).hexdigest(),
    workbook_rows=len(raw), valid_protein_variants=int(raw.variant_short.notna().sum()),
    unparsed_measurements=nonparsed_examples[:30], assay_status_counts=status_counts,
    safe_candidate_rows=len(safe), safe_candidate_positions=int(safe.protein_position.nunique()),
    security_note='Only auxiliary external labels; per-fold held-out VTI protein positions must still be removed.'),indent=2,default=int)+'\n')
print('Saved:',meta)
print('PASS: External-data coverage and cross-study position overlap audited; no protected VTI outcomes opened.')
