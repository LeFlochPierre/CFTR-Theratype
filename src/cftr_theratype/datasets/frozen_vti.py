"""Canonical TRAIN-only VTI loader; no protected VAL/TEST outcomes opened."""
from pathlib import Path
import numpy as np
import pandas as pd
from cftr_theratype import model_zoo
from cftr_theratype.evaluation.folds import from_historical_oof

ASSAY_COLS = [
    'variant_key', 'baseline_fsk_pct_wt', 'iva_treated_pct_wt',
    'vnz_tez_fsk_pct_wt', 'vti_treated_pct_wt', 'vti_rescue_pp',
    'corrector_rescue_pp',
]


def load_train(root=None):
    root = Path.cwd() if root is None else Path(root)
    base, features, historical, _ = model_zoo.prepare(root)
    base = base.copy()
    if (len(base) != 210 or not base.variant_key.is_unique
            or base.protein_position.nunique() != 168
            or not base.transfer_split.eq('train').all()):
        raise ValueError('Frozen VTI TRAIN cohort mismatch')
    if not {'sequence', 'sequence_structure'} <= set(features):
        raise ValueError('Missing frozen mutation representations')
    for name in ('sequence', 'sequence_structure'):
        X = np.asarray(features[name], dtype=float)
        if X.ndim != 2 or X.shape[0] != len(base):
            raise ValueError(f'{name}: invalid shape {X.shape}')
        if np.isinf(X).any():
            X = X.copy()
            X[np.isinf(X)] = np.nan
        features[name] = X
    keys = base.variant_key.astype(str).tolist()
    assay = pd.read_parquet(
        root / 'data/processed/bihler_2026_vti.parquet',
        columns=ASSAY_COLS, filters=[('variant_key', 'in', keys)],
    )
    if not assay.variant_key.is_unique or set(assay.variant_key) != set(keys):
        raise ValueError('Frozen VTI TRAIN assay membership mismatch')
    d = base.merge(assay, on='variant_key', how='left',
                   validate='one_to_one', suffixes=('', '_assay'))
    if len(d) != 210 or d.variant_key.astype(str).tolist() != keys:
        raise ValueError('Frozen feature/target row order mismatch')
    for col in ASSAY_COLS[1:]:
        d[col] = pd.to_numeric(d[col], errors='raise')
        if not np.isfinite(d[col].to_numpy(float)).all():
            raise ValueError(f'Nonfinite VTI TRAIN assay target: {col}')
    def consistent(a, b):
        return np.allclose(a, b, atol=1e-6)
    if not consistent(d.vti_rescue_pp, d.vti_treated_pct_wt-d.baseline_fsk_pct_wt):
        raise ValueError('Total rescue identity failed')
    if not consistent(d.corrector_rescue_pp,
                      d.vnz_tez_fsk_pct_wt-d.baseline_fsk_pct_wt):
        raise ValueError('Corrector rescue identity failed')
    if not consistent(d.vti_rescue_pp, base.vti_rescue_pp):
        raise ValueError('VTI base target mismatch')
    d['vti_acute_after_corrector_pp'] = (
        d.vti_treated_pct_wt - d.vnz_tez_fsk_pct_wt
    )
    if not consistent(d.vti_rescue_pp,
                      d.corrector_rescue_pp+d.vti_acute_after_corrector_pp):
        raise ValueError('Component decomposition failed')
    positions = dict(zip(d.variant_key, d.protein_position))
    splits, reference = from_historical_oof(historical, keys, positions)
    if len(splits) != 10:
        raise ValueError(f'Expected ten frozen outer splits, got {len(splits)}')
    return d, features, historical, splits, reference
