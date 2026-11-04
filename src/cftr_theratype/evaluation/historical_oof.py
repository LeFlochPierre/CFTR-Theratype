"""Load and verify historical TRAIN-only Phase 26G / 31I M0 predictions.

This is a parity checker, not an instruction to inspect protected VAL/TEST
outcomes. The root must contain already-existing frozen OOF files.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assert_oof_valid(oof,d,kind):
    needed={'repeat','fold','variant_key','protein_position','observed','prediction'}
    if not needed.issubset(oof):
        raise ValueError(f'{kind}: missing columns {needed-set(oof)}')
    expected=set(d.variant_key.astype(str))
    positions=d.set_index('variant_key').protein_position.astype(int).to_dict()
    outcomes=dict(zip(d.variant_key.astype(str), d.corrector_rescue_pp if kind=='corrector' else d.vti_rescue_pp))
    if not np.isfinite(oof[['observed','prediction']].to_numpy(float)).all():
        raise AssertionError(f'{kind}: nonfinite OOF observations/predictions')
    for repeat,x in oof.groupby('repeat'):
        if len(x)!=len(d) or x.variant_key.duplicated().any() or set(x.variant_key)!=expected:
            raise AssertionError(f'{kind}: repeat {repeat} incomplete or duplicated')
        for row in x.itertuples(index=False):
            if int(row.protein_position)!=positions[str(row.variant_key)]:
                raise AssertionError(f'{kind}: position mismatch')
            if not np.isclose(float(row.observed),float(outcomes[str(row.variant_key)]),atol=1e-5):
                raise AssertionError(f'{kind}: unexpected observed TRAIN target')
    if set(oof.repeat)!= {1,2} or set(oof.fold)!={1,2,3,4,5}:
        raise AssertionError(f'{kind}: expected two historical repeats, five folds')
    if oof.duplicated(['repeat','fold','variant_key']).any():
        raise AssertionError(f'{kind}: duplicated fold membership')


def load_m0(d, root=None):
    ROOT = Path.cwd() if root is None else Path(root)
    samples=[]; inputs={}
    for target in ('corrector','total'):
        phase26=ROOT/f'results/biology/phase26g/phase26g_{target}_oof.csv'
        phase31=ROOT/'results/biology/phase31i/phase31i2_repo_oof.csv'
        for p in (phase26,phase31):
            if not p.is_file():
                raise FileNotFoundError(f'Missing frozen OOF file: {p}')
            inputs[str(p.relative_to(ROOT))]=digest(p)
        past=pd.read_csv(phase26,usecols=['repeat','fold','variant_key','protein_position','observed','xgb_80'])
        past=past.rename(columns={'xgb_80':'prediction'})
        audit=pd.read_csv(phase31,usecols=['target','model','repeat','fold','variant_key','protein_position','observed','prediction'])
        check=audit.loc[(audit.target==target)&(audit.model=='M0')].drop(columns=['target','model'])
        assert_oof_valid(past,d,target)
        assert_oof_valid(check,d,target)
        matched=past.merge(check,on=['repeat','fold','variant_key','protein_position'],how='outer',indicator=True,
                           suffixes=('_26g','_31i'),validate='one_to_one')
        if len(matched)!=len(past) or not matched._merge.eq('both').all():
            raise AssertionError('Phase26G and Phase31I M0 row misalignment')
        if not np.allclose(matched.observed_26g,matched.observed_31i,atol=1e-7):
            raise AssertionError('Phase26G and Phase31I M0 response mismatch')
        max_diff=float(np.max(np.abs(matched.prediction_26g-matched.prediction_31i)))
        if max_diff>1e-4:
            raise AssertionError(f'Historical M0 parity failed: {max_diff}')
        print(f'M0 parity [{target}]: max_abs={max_diff:.8g}; rows={len(past)}')
        past['target']=target
        past['model']='M0'
        samples.append(past)
    return pd.concat(samples,ignore_index=True), inputs


