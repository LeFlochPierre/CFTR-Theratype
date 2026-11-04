"""Frozen TRAIN-only Phase31M/32A input alignment and ESMRank score loading.

Shared inputs; historical run-specific model fitting stays in archived workflows.
No VTI validation/test response values are opened by these functions.
"""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from cftr_theratype.datasets.frozen_vti import load_train as load_vti_train
from cftr_theratype.datasets.assay_identifiers import with_ids, norm_aa


def get_data(root):
    root = Path(root)
    d, feat, _, folds, source = load_vti_train(root)
    X = np.asarray(feat['sequence_structure'], dtype=float)
    if len(d) != 210 or d.protein_position.nunique() != 168 or X.shape != (210, 80):
        raise AssertionError('Unexpected frozen VTI TRAIN or feature representation')
    if not d.transfer_split.eq('train').all() or np.isinf(X).any():
        raise AssertionError('Protected partitions or invalid features')
    keys = d.variant_key.astype(str).tolist()
    metadata = pd.read_parquet(root/'data/processed/bihler_2026_transfer_splits.parquet',
                               columns=['variant_key','hgvs_protein','protein_position','transfer_split'])
    metadata = metadata[metadata.transfer_split.eq('train')]
    v = d[['variant_key','protein_position']].merge(
        metadata[['variant_key','hgvs_protein','protein_position']], on='variant_key',
        how='left', validate='one_to_one',suffixes=('','_meta'))
    if v.hgvs_protein.isna().any() or not v.protein_position.eq(v.protein_position_meta).all():
        raise AssertionError('VTI metadata/position mismatch')
    v = with_ids(v, 'VTI TRAIN', strict=True)
    if not v.aa31i.str.fullmatch(r'[ACDEFGHIKLMNPQRSTVWY]\d+[ACDEFGHIKLMNPQRSTVWY]').all():
        raise AssertionError('Non-missense VTI variant: ESMRank and multitask need separate handling')
    if list(v.variant_key.astype(str)) != keys:
        raise AssertionError('Variant order mismatch')
    return d, X, folds, v


def load_esm(path):
    z=pd.read_csv(path)
    if not {'ids','prediction'}.issubset(z): raise ValueError('Expected ESMRank predictions columns: ids,prediction')
    z=z[['ids','prediction']].copy()
    z['aa31i']=z.ids.astype(str).map(norm_aa)
    if z.aa31i.isna().any() or z.aa31i.duplicated().any():
        raise ValueError('Cannot map unique ESMRank ids to substitutions; inspect output')
    z.prediction=pd.to_numeric(z.prediction,errors='coerce')
    if not np.isfinite(z.prediction.to_numpy(float)).all():raise ValueError('Nonfinite ESMRank score')
    return z.set_index('aa31i').prediction
