"""Exact Phase29A position-weighted diagnostics and repeated-OOF helpers.

No disk I/O, model fitting, or protected assay outcomes are performed here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cftr_theratype.models.response_capacity import CONFIGS


def pos_mae(d, label, pred):
    q=d[['protein_position',label,pred]].copy()
    q['ae']=(q[label]-q[pred]).abs()
    return float(q.groupby('protein_position').ae.mean().mean())


def gain_bootstrap(frame, candidate, reference, truth, nboot, seed):
    q=frame[['repeat','protein_position',candidate,reference,truth]].copy()
    q['delta']=(q[reference]-q[truth]).abs()-(q[candidate]-q[truth]).abs()
    perpos=(q.groupby(['repeat','protein_position']).delta.mean()
            .groupby('protein_position').mean().to_numpy(float))
    rng=np.random.default_rng(seed)
    draws=rng.integers(0,len(perpos),size=(nboot,len(perpos)))
    dist=perpos[draws].mean(axis=1)
    return {'target':truth,'candidate':candidate,'reference':reference,
            'mae_gain_pp':float(perpos.mean()),
            'ci95_low':float(np.quantile(dist,.025)),
            'ci95_high':float(np.quantile(dist,.975)),
            'positions':len(perpos)}


def averaged_one_variant(oof):
    first=oof.groupby(['variant_key','protein_position'],as_index=False).agg(
        observed_corrector=('observed_corrector','first'),
        observed_total=('observed_total','first'))
    preds=oof.groupby('variant_key',as_index=False)[[f'corrector_{k}' for k in CONFIGS]].mean()
    out=first.merge(preds,on='variant_key',validate='one_to_one')
    if len(out)!=210: raise ValueError('Expected 210 variants after repeat aggregation')
    return out
