"""Report *existing frozen TRAIN OOF* results without retraining or reading VAL/TEST.

This is a development benchmark summary, never a claim of external generalisation.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from cftr_theratype.evaluation.metrics import position_mae


def generate(root, *, output=None):
    root = Path(root).resolve()
    records = []
    for target in ('total', 'corrector'):
        file = root / f'results/biology/phase26g/phase26g_{target}_oof.csv'
        if not file.is_file():
            raise FileNotFoundError(f'Frozen TRAIN benchmark missing: {file}')
        d = pd.read_csv(file, dtype={'variant_key': str})
        needed = {'repeat', 'fold', 'variant_key', 'protein_position', 'observed', 'xgb_80'}
        if needed - set(d):
            raise ValueError(f'Missing frozen OOF columns: {needed - set(d)}')
        if len(d) != 420 or d.duplicated(['repeat', 'variant_key']).any() or d.groupby('repeat').variant_key.nunique().to_dict() != {1: 210, 2: 210}:
            raise ValueError(f'Invalid frozen OOF membership for {target}')
        models = [c for c in ('xgb_80', 'ridge_80', 'extratrees_80', 'xgb_selected32_80') if c in d]
        for model in models:
            mae = []
            for rep, g in d.groupby('repeat'):
                y = g.observed.to_numpy(float)
                pred = g[model].to_numpy(float)
                if not np.isfinite(y).all() or not np.isfinite(pred).all():
                    raise ValueError(f'Nonfinite {target}/{model}')
                mae.append(position_mae(y, pred, g.protein_position.to_numpy()))
            records.append((target, model, float(np.mean(mae))))
    lines = ['# CFTR-Theratype development benchmark', '',
             'Frozen VTI TRAIN-only, 210 variants / 168 positions, 2 × 5 position-disjoint folds.',
             'Repeated CV folds are not independent test cohorts.',
             'These exploratory scores are NOT independent clinical validation.', '',
             '| Target | Model | Mean position MAE (pp) |', '|---|---|---:|']
    for target, model, value in records:
        lines.append(f'| {target} | {model} | {value:.4f} |')
    result = '\n'.join(lines) + '\n'
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f'Refusing to overwrite report: {output}')
        output.write_text(result)
        print('Report written:', output)
    else:
        print(result)
    return result
