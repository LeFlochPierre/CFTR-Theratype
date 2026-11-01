#!/usr/bin/env python3
"""Phase 23A — ETI functional auxiliary-data audit for CFTR-Theratype.

Purpose
-------
Count safe Bihler 2024 ETI training examples and available acute-IVA
functional labels. Compare against VTI *TRAIN* acute-IVA responses only.
This is an audit, not an ML model; no protected outcomes are opened.

Inputs: frozen partition metadata + condition-paired ETI/VTI Parquet.
Safety:
 * Only frozen 2024 split=train variants' outcomes are scanned via Arrow.
 * Only frozen 2026 VTI transfer_split=train variants' outcomes scanned.
 * Exclude 2024 candidate positions matching 2026 val/test/excluded positions.
 * Exclude 2024 candidate variant keys matching 2026 val/test/excluded keys,
   including positions that could not be parsed.
 * Additional outer/inner held-out VTI position filtering remains REQUIRED in
   future predictive models; output is only a candidate auxiliary pool.

Run:
  python scripts/phase23a_eti_functional_overlap_audit.py --self-test
  python scripts/phase23a_eti_functional_overlap_audit.py | tee logs/phase23a_eti_audit.log
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for buf in iter(lambda: f.read(1024 * 1024), b''):
            h.update(buf)
    return h.hexdigest()


def require(c, m):
    if not bool(c):
        raise AssertionError(m)


def count_distribution(frame: pd.DataFrame, label: str, tag: str):
    z = frame[[label, 'protein_position']].dropna().copy()
    z[label] = pd.to_numeric(z[label], errors='coerce')
    z = z.loc[np.isfinite(z[label])].copy()
    if not len(z):
        print(f'{tag}: no finite outcomes')
        return {'n': 0, 'positions': 0}
    summaries = {
        'n': int(len(z)), 'positions': int(z.protein_position.nunique()),
        'mean_pp': float(z[label].mean()),
        'median_pp': float(z[label].median()),
        'sd_pp': float(z[label].std()),
        'min_pp': float(z[label].min()),
        'max_pp': float(z[label].max()),
        'gt20_variants': int((z[label] > 20).sum()),
        'gt20_positions': int(z.loc[z[label] > 20, 'protein_position'].nunique()),
        'gt40_variants': int((z[label] > 40).sum()),
        'gt40_positions': int(z.loc[z[label] > 40, 'protein_position'].nunique()),
        'lte2_variants': int((z[label] <= 2).sum()),
    }
    print(f"{tag:45s} n={summaries['n']:3d}  positions={summaries['positions']:3d}"
          f"  mean={summaries['mean_pp']:8.3f}  median={summaries['median_pp']:8.3f}"
          f"  >20={summaries['gt20_variants']:3d} ({summaries['gt20_positions']} positions)"
          f"  >40={summaries['gt40_variants']:3d} ({summaries['gt40_positions']} positions)")
    return summaries


def filter_aux_for_heldout(aux: pd.DataFrame, heldout_positions) -> pd.DataFrame:
    """MUST be repeated for inner AND outer held-out position sets in ML."""
    blocked = set(map(int, heldout_positions))
    if 'protein_position' not in aux:
        raise ValueError('aux missing protein_position')
    return aux.loc[~aux['protein_position'].astype(int).isin(blocked)].copy()


def read_labeled_filtered(path: Path, permitted_keys: set[str], columns: list[str]) -> pd.DataFrame:
    """Never load all outcome rows before filtering by TRAIN-only keys."""
    require(bool(permitted_keys), f'No permitted keys for {path}')
    import pyarrow.dataset as ds
    return ds.dataset(str(path), format='parquet').to_table(
        columns=columns,
        filter=ds.field('variant_key').isin(sorted(permitted_keys))
    ).to_pandas()


def audit(root: Path):
    d = root / 'data/processed'
    paths = {
        'old_split': d / 'bihler_2024_missense_splits.parquet',
        'old_assay': d / 'bihler_2024_theratyping.parquet',
        'vti_membership': d / 'bihler_2026_transfer_splits.parquet',
        'vti_assay': d / 'bihler_2026_vti.parquet',
    }
    for name, p in paths.items():
        require(p.is_file(), f'Missing {name}: {p}')

    print('='*106)
    print('PHASE 23A | ETI 2024 FUNCTIONAL SUPERVISION / FROZEN POSITION OVERLAP')
    print('='*106)
    print('Strict scope: 2024 frozen TRAIN outcomes + 2026 VTI frozen TRAIN outcomes ONLY')
    print('2024 VAL/TEST and 2026 VAL/TEST/EXCLUDED labels never requested')
    print('External ETI outcomes are different treatments; never pool as VTI labels')

    old = pd.read_parquet(paths['old_split'])  # membership only
    vti = pd.read_parquet(paths['vti_membership'])  # membership only
    require(len(old) == 586, f'Unexpected Bihler 2024 missense membership count {len(old)}')
    require(len(vti) == 415, f'Unexpected VTI membership count {len(vti)}')
    for name, t, k, c in [
        ('old', old, 'variant_key', 'split'),
        ('vti', vti, 'variant_key', 'transfer_split')]:
        require(not t[k].isna().any(), f'{name}: null keys')
        require(t[k].astype(str).is_unique, f'{name}: duplicate keys')
        require(c in t and 'protein_position' in t, f'{name}: missing membership column')
        t[k] = t[k].astype(str)
        t[c] = t[c].astype(str).str.strip().str.lower()
        t['protein_position'] = pd.to_numeric(t['protein_position'], errors='coerce')

    require(set(old.split.unique()) == {'train', 'val', 'test'},
            f'Unexpected 2024 splits {old.split.value_counts().to_dict()}')
    require(set(vti.transfer_split.unique()) == {'train', 'val', 'test', 'excluded'},
            f'Unexpected VTI splits {vti.transfer_split.value_counts().to_dict()}')
    for split, expected in [('train', 210), ('val', 52), ('test', 33), ('excluded', 120)]:
        require(int((vti.transfer_split == split).sum()) == expected,
                f'Unexpected 2026 {split} count')
    for s in ('train', 'val', 'test'):
        require(old.loc[old.split.eq(s), 'protein_position'].notna().all(),
                f'2024 {s}: unknown positions; stop to protect splits')
    for s in ('train', 'val', 'test'):
        require(vti.loc[vti.transfer_split.eq(s), 'protein_position'].notna().all(),
                f'2026 {s}: unknown positions; stop to protect splits')
    old['protein_position'] = old.protein_position.astype(int)
    old_sets = {s: set(old.loc[old.split.eq(s), 'protein_position'])
                for s in ('train', 'val', 'test')}
    for a in old_sets:
        for b in old_sets:
            if a < b:
                require(old_sets[a].isdisjoint(old_sets[b]),
                        f'2024 {a}/{b} positions overlap')
    vti_sets = {s: set(vti.loc[vti.transfer_split.eq(s), 'protein_position'].dropna().astype(int))
                for s in ('train', 'val', 'test', 'excluded')}
    for a in vti_sets:
        for b in vti_sets:
            if a < b:
                require(vti_sets[a].isdisjoint(vti_sets[b]),
                        f'2026 {a}/{b} positions overlap')

    print('\nA. FROZEN MEMBERSHIPS (metadata only)')
    print('2024 missense:')
    print(old.groupby('split').agg(variants=('variant_key', 'nunique'),
                                   positions=('protein_position', 'nunique')).to_string())
    print('2026 transfer:')
    print(vti.groupby('transfer_split').agg(variants=('variant_key', 'nunique'),
                                            positions=('protein_position', 'nunique')).to_string())
    missing_excl = int(vti.loc[vti.transfer_split.eq('excluded'), 'protein_position'].isna().sum())
    print(f'VTI EXCLUDED rows without parseable position: {missing_excl}; keys also protected')

    old_train = old.loc[old.split.eq('train')].copy()
    vti_train = vti.loc[vti.transfer_split.eq('train')].copy()
    train_keys = set(vti_train.variant_key)
    protected_vti = vti.loc[~vti.transfer_split.eq('train')].copy()
    protected_keys = set(protected_vti.variant_key)
    protected_positions = (vti_sets['val'] | vti_sets['test'] | vti_sets['excluded'] |
                           old_sets['val'] | old_sets['test'])
    all_vti_positions = set().union(*vti_sets.values())

    old_train['blocked_by_vti_position'] = old_train.protein_position.isin(protected_positions)
    old_train['blocked_by_vti_key'] = old_train.variant_key.isin(protected_keys)
    old_train['allowed_for_etivti_aux'] = ~(
        old_train.blocked_by_vti_position | old_train.blocked_by_vti_key)
    candidates_meta = old_train.loc[old_train.allowed_for_etivti_aux].copy()
    require(not (set(candidates_meta.protein_position) & protected_positions),
            'Protected 2026 or old 2024 positions in auxiliary candidate list')
    require(not (set(candidates_meta.variant_key) & protected_keys),
            'Protected VTI keys in auxiliary candidate list')
    require(not candidates_meta.variant_key.duplicated().any(), 'Duplicate ETI candidate key')
    candidates_meta['position_relation_to_vti'] = np.where(
        candidates_meta.protein_position.isin(vti_sets['train']),
        'at_vti_train_position',
        'novel_to_vti',
    )
    require(not candidates_meta.protein_position.isin(vti_sets['val'] | vti_sets['test'] |
                                                     vti_sets['excluded']).any(),
            'Novel/allowed position overlaps protected VTI')

    print('\nB. 2024 TRAIN CANDIDATE FILTERING (metadata only)')
    print(f'Old 2024 TRAIN: {len(old_train)} variants / {old_train.protein_position.nunique()} positions')
    print('Blocked by protected VTI/old-split position:', int(old_train.blocked_by_vti_position.sum()))
    print('Blocked by protected VTI variant key:', int(old_train.blocked_by_vti_key.sum()))
    print('Union blocked:', int((~old_train.allowed_for_etivti_aux).sum()))
    print('SAFE initial 2024 ETI candidate rows:', len(candidates_meta))
    print(candidates_meta.groupby('position_relation_to_vti').agg(
        variants=('variant_key', 'nunique'), positions=('protein_position', 'nunique')).to_string())
    print('WARNING: This is NOT final outer/inner-fold-safe auxiliary membership')

    # Important: build keys FIRST; only then scan labeled Parquet filtered to those keys.
    eti_keys = set(candidates_meta.variant_key)
    eti_cols = [
        'variant_key', 'hgvs_protein', 'baseline_fsk_pct_wt',
        'iva_treated_pct_wt', 'elx_tez_fsk_pct_wt', 'eti_treated_pct_wt',
        'iva_rescue_pp', 'corrector_rescue_pp', 'eti_rescue_pp',
    ]
    eti = read_labeled_filtered(paths['old_assay'], eti_keys, eti_cols)
    require(len(eti) == len(candidates_meta) and eti.variant_key.is_unique,
            '2024 paired assay did not cover exactly all permitted ETI candidates')
    eti = candidates_meta[['variant_key','protein_position','position_relation_to_vti',
                            'hgvs_protein']].rename(columns={'hgvs_protein':'frozen_hgvs_protein'}).merge(
        eti, on='variant_key', how='left', validate='one_to_one')
    assert len(eti) == len(candidates_meta)
    fcols = [c for c in eti_cols if c not in ('variant_key','hgvs_protein')]
    for c in fcols:
        eti[c] = pd.to_numeric(eti[c], errors='coerce')
    eti['eti_acute_iva_pp'] = eti.eti_treated_pct_wt - eti.elx_tez_fsk_pct_wt
    eti['dmso_acute_iva_pp'] = eti.iva_treated_pct_wt - eti.baseline_fsk_pct_wt
    eti['eti_corrector_fsk_gain_pp'] = eti.elx_tez_fsk_pct_wt - eti.baseline_fsk_pct_wt
    eti['eti_total_fsk_iva_rescue_pp'] = eti.eti_treated_pct_wt - eti.baseline_fsk_pct_wt
    eti['delta_acute_iva_eti_minus_dmso_pp'] = eti.eti_acute_iva_pp - eti.dmso_acute_iva_pp
    require(np.isfinite(eti[['eti_acute_iva_pp','dmso_acute_iva_pp']]).all().all(),
            'Some ETI/DMSO functional increments are missing or nonfinite')
    # Protocol check: do not conflate the two meanings of "rescue".
    rescue_difference = (eti.iva_rescue_pp - eti.dmso_acute_iva_pp).abs()
    print('\nC. ALLOWED ETI FUNCTIONAL LABEL DISTRIBUTIONS (TRAIN only)')
    all_results = {}
    for label, tag in [
        ('eti_acute_iva_pp', 'ETI-corrected acute IVA increment'),
        ('dmso_acute_iva_pp', 'DMSO acute IVA increment'),
        ('eti_corrector_fsk_gain_pp', 'ELX+TEZ FSK gain'),
        ('eti_total_fsk_iva_rescue_pp', 'Total ETI-treated versus DMSO FSK gain')]:
        all_results[label] = count_distribution(eti, label, tag)
    print('Legacy 2024 iva_rescue_pp vs calculated DMSO acute difference: '
          f'median |delta|={rescue_difference.median():.6f}; '
          f'max |delta|={rescue_difference.max():.6f}')
    if rescue_difference.max() > 0.01:
        print('WARNING: Raw derived DMSO increment differs from legacy iva_rescue_pp; review semantics')
    for group, subset in eti.groupby('position_relation_to_vti'):
        print(f'\nETI subset: {group}')
        count_distribution(subset, 'eti_acute_iva_pp', 'ETI-corrected acute IVA')
        count_distribution(subset, 'dmso_acute_iva_pp', 'DMSO acute IVA')

    # VTI TRAIN labels only: read VTI assay with Arrow filtered by known train IDs.
    vti_cols = ['variant_key', 'hgvs_protein', 'vnz_tez_fsk_pct_wt',
                'vti_treated_pct_wt','baseline_fsk_pct_wt','iva_treated_pct_wt']
    vti_lab = read_labeled_filtered(paths['vti_assay'], train_keys, vti_cols)
    require(len(vti_lab) == 210 and vti_lab.variant_key.is_unique,
            'VTI TRAIN-only assay lookup did not return 210 unique rows')
    vti_lab = vti_train[['variant_key','protein_position']].merge(
        vti_lab, on='variant_key', how='left', validate='one_to_one')
    vti_lab['vti_acute_iva_pp'] = (vti_lab.vti_treated_pct_wt -
                                  vti_lab.vnz_tez_fsk_pct_wt)
    vti_lab['vti_dmso_iva_pp'] = (vti_lab.iva_treated_pct_wt -
                                 vti_lab.baseline_fsk_pct_wt)
    require(np.isfinite(vti_lab.vti_acute_iva_pp).all(), 'Missing VTI train labels')

    # Exact nucleotide variant matches, then same-HGVS-only matches (descriptive).
    matched = eti.merge(
        vti_lab[['variant_key','vti_acute_iva_pp','vti_dmso_iva_pp','hgvs_protein']],
        on='variant_key', how='inner', validate='one_to_one', suffixes=('_eti','_vti'))
    require((matched.hgvs_protein_eti.astype(str) == matched.hgvs_protein_vti.astype(str)).all(),
            'ETI/VTI exact nucleotide match with discordant HGVS protein')
    matched['eti_minus_vti_acute_pp'] = (
        matched.eti_acute_iva_pp - matched.vti_acute_iva_pp)
    print('\nD. EXACT ETI/VTI TRAIN SUBSTITUTION MATCHES')
    print('Exact variant_key matches:', len(matched))
    print('Distinct positions:', matched.protein_position.nunique() if len(matched) else 0)
    if len(matched) >= 3:
        for col in ('eti_acute_iva_pp','dmso_acute_iva_pp'):
            corr = matched[[col,'vti_acute_iva_pp']].corr(method='spearman').iloc[0,1]
            print(f'{col:22s} vs VTI acute increment: Spearman rho={corr:+.3f}')
        print('ETI acute minus VTI acute: median=%+.3f pp; median absolute=%0.3f pp' % (
            matched.eti_minus_vti_acute_pp.median(), matched.eti_minus_vti_acute_pp.abs().median()))
        print('Cross-study >20-pp agreement table (same mutation, different corrector):')
        print(pd.crosstab(matched.eti_acute_iva_pp > 20,
                          matched.vti_acute_iva_pp > 20,
                          rownames=['ETI acute >20'], colnames=['VTI acute >20']).to_string())
    else:
        print('Insufficient matches for meaningful cross-assay correlation')
    print('WARNING: Matched ETI/VTI labels are not exchangeable or independent replicates')

    # Outer-fold safeguard demonstration: test position-blocking and inspect totals.
    print('\nE. AUXILIARY LEAKAGE CHECK / FUTURE CV REQUIREMENTS')
    print('Initial auxiliary positions:', eti.protein_position.nunique())
    test_held = sorted(vti_sets['train'])[:8]
    demonstration = filter_aux_for_heldout(eti, test_held)
    require(not (set(demonstration.protein_position) & set(test_held)),
            'Fold filtering helper failed')
    print(f'Fold filter self-check: blocked {len(test_held)} example VTI-train positions; '
          f'candidate variants remaining={len(demonstration)}; PASS')
    print('In Phase 23B: exclude external ETI rows at held-out VTI positions in every')
    print('outer fold AND each grouped inner-selection fold. ETI 2024 val/test rows are never eligible.')
    print('Study/treatment are task identifiers; do not treat them as VTI response labels.')

    out = root / 'results/biology/phase23a'
    out.mkdir(parents=True, exist_ok=True)
    eti_out = out / 'eti2024_train_auxiliary_candidates.csv'
    matched_out = out / 'eti_vti_exact_train_matches.csv'
    metadata_out = out / 'eti2024_training_position_eligibility.csv'
    report_out = out / 'phase23a_report.json'
    eti.to_csv(eti_out, index=False)
    matched.to_csv(matched_out, index=False)
    old_train.to_csv(metadata_out, index=False)
    report = {
        'protocol': '2024 frozen TRAIN labels + 2026 frozen VTI TRAIN labels only',
        'sha256': {k: sha256(p) for k,p in paths.items()},
        '2024_frozen_train_variants': int(len(old_train)),
        '2024_frozen_train_positions': int(old_train.protein_position.nunique()),
        '2024_train_candidates': int(len(eti)),
        '2024_train_candidate_positions': int(eti.protein_position.nunique()),
        'vti_train_position_candidate_variants': int((eti.position_relation_to_vti=='at_vti_train_position').sum()),
        'novel_to_vti_candidate_variants': int((eti.position_relation_to_vti=='novel_to_vti').sum()),
        'novel_to_vti_candidate_positions': int(eti.loc[eti.position_relation_to_vti=='novel_to_vti','protein_position'].nunique()),
        'exact_training_variant_matches': int(len(matched)),
        'unparseable_vti_excluded_membership_position_count': missing_excl,
        'eti_label_summaries': all_results,
        'warning': 'Auxiliary candidates MUST be filtered anew at each outer and inner held-out VTI protein position',
    }
    report_out.write_text(json.dumps(report, indent=2, allow_nan=False))
    print('\nOUTPUTS:')
    for x in (eti_out, matched_out, metadata_out, report_out):
        print(' ', x.relative_to(root))
    print('PASS: Phase23A complete. Frozen 2024 and VTI protected outcomes unopened.')


def self_test():
    a = pd.DataFrame({'protein_position': [10,10,11,12], 'value':[1,2,3,4]})
    assert filter_aux_for_heldout(a,{10,12}).value.tolist() == [3]
    assert filter_aux_for_heldout(a,{11}).value.tolist() == [1,2,4]
    assert filter_aux_for_heldout(a,set()).equals(a)
    for z in ({10,12},{11},set()):
        assert set(filter_aux_for_heldout(a,z).protein_position).isdisjoint(z)
    print('PASS: Held-out protein-position filtering self-test')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path.cwd())
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()
    if args.self_test:
        self_test()
        return
    audit(args.root.resolve())


if __name__ == '__main__':
    main()
