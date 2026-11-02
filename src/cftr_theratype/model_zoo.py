"""Phase 17A: nested, position-disjoint model comparison on frozen VTI TRAIN.

Call: python -m cftr_theratype.model_zoo [--smoke]
Reuses Phase 15C + 16C validated input provenance; never reads VTI/ETI
validation or test response outcomes. Outer folds EXACTLY match the original
2x5 OOF assignment. Inner folds are group-disjoint and fitted from scratch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor

from cftr_theratype import stability_benchmark as sb
from cftr_theratype import structure_benchmark as base

SEED = 20261023
N_BOOT = 2500
FEATURE_SETS = ('sequence', 'sequence_structure', 'sequence_structure_thermompnn')
FAMILIES = ('catboost', 'xgboost', 'extra_trees', 'rbf_svr', 'elastic_net', 'gaussian_process')
REFERENCES = (sb.MODEL_SEQUENCE, sb.MODEL_STRUCTURE, sb.MODEL_STABILITY)
THRESHOLD = 10.0


def candidates(family: str) -> tuple[dict, ...]:
    """Small, explicitly predeclared grids (not a large adaptive search)."""
    grids = {
        'catboost': ({'depth': 2}, {'depth': 3}, {'depth': 5}),
        'xgboost': ({'max_depth': 2}, {'max_depth': 3}, {'max_depth': 4}),
        'extra_trees': ({'min_samples_leaf': 2}, {'min_samples_leaf': 4}, {'min_samples_leaf': 8}),
        'rbf_svr': ({'C': 1.0}, {'C': 10.0}, {'C': 100.0}),
        'elastic_net': ({'alpha': 0.1}, {'alpha': 1.0}, {'alpha': 10.0}),
        'gaussian_process': ({'length_scale': 4.0}, {'length_scale': 12.0}),
    }
    if family not in grids:
        raise ValueError(f'Unrecognised model family {family!r}')
    return grids[family]


class FoldRegressor:
    """Explicit fold-fitted imputation / scaling, CatBoost-compatible across sklearn versions."""
    def __init__(self, model, *, scale: bool):
        self.model = model
        self.imputer = SimpleImputer(strategy='median', keep_empty_features=True)
        self.scaler = StandardScaler() if scale else None
        self.fitted = False

    def fit(self, X, y):
        X = self.imputer.fit_transform(X)
        if self.scaler is not None:
            X = self.scaler.fit_transform(X)
        self.model.fit(X, y)
        self.fitted = True
        return self

    def predict(self, X):
        if not self.fitted:
            raise ValueError('Regressor must be fitted on training fold before prediction')
        X = self.imputer.transform(X)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.model.predict(X)


def estimator(family: str, params: dict, seed: int):
    """Build fresh fold estimator; preprocessing is fitted INSIDE folds."""
    if family == 'catboost':
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(iterations=250, depth=params['depth'], learning_rate=.04,
                                  l2_leaf_reg=8, loss_function='RMSE', random_seed=seed,
                                  thread_count=1, allow_writing_files=False, verbose=False)
    elif family == 'xgboost':
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=250, max_depth=params['max_depth'], learning_rate=.035,
                             min_child_weight=5, reg_lambda=12., subsample=.85,
                             colsample_bytree=.85, tree_method='hist', objective='reg:squarederror',
                             random_state=seed, n_jobs=1, verbosity=0)
    elif family == 'extra_trees':
        model = ExtraTreesRegressor(n_estimators=200, min_samples_leaf=params['min_samples_leaf'],
                                    max_features=.75, n_jobs=1, random_state=seed)
    elif family == 'rbf_svr':
        model = SVR(C=params['C'], epsilon=5., kernel='rbf', gamma='scale')
    elif family == 'elastic_net':
        model = ElasticNet(alpha=params['alpha'], l1_ratio=.5, max_iter=20000,
                           selection='cyclic', random_state=seed)
    elif family == 'gaussian_process':
        kernel = ConstantKernel(1., constant_value_bounds='fixed') * RBF(
            length_scale=params['length_scale'], length_scale_bounds='fixed') + WhiteKernel(
                noise_level=1., noise_level_bounds='fixed')
        model = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=True,
                                         alpha=1e-6, copy_X_train=True, random_state=seed)
    else:
        raise ValueError(family)
    return FoldRegressor(model, scale=family in {'rbf_svr', 'elastic_net', 'gaussian_process'})


def position_mae(y: np.ndarray, pred: np.ndarray, positions: np.ndarray) -> float:
    """Equal protein-position weighting, regardless of variant multiplicity."""
    err = np.abs(np.asarray(y, float) - np.asarray(pred, float))
    pos = np.asarray(positions)
    if len(err) != len(pos) or len(err) == 0 or not np.isfinite(err).all():
        raise ValueError('Invalid position-MAE inputs')
    return float(pd.DataFrame({'pos': pos, 'error': err}).groupby('pos').error.mean().mean())


def inner_select(X: np.ndarray, y: np.ndarray, positions: np.ndarray, family: str,
                 seed: int, n_inner: int = 3) -> tuple[dict, float]:
    """Select hyperparameters solely using inner TRAIN-position-held-out labels."""
    X, y, positions = np.asarray(X, float), np.asarray(y, float), np.asarray(positions)
    if X.ndim != 2 or len(X) != len(y) or len(y) != len(positions):
        raise ValueError('Invalid nested-training arrays')
    if not np.isfinite(y).all() or np.isinf(X).any():
        raise ValueError('Nonfinite nested-training arrays')
    if len(np.unique(positions)) < n_inner:
        raise ValueError('Too few distinct positions for inner CV')
    inner = list(GroupKFold(n_splits=n_inner).split(X, y, groups=positions))
    verify_splits(inner, positions)
    best, score = None, np.inf
    for settings in candidates(family):
        predictions = np.full(len(y), np.nan, float)
        for fold, (f, h) in enumerate(inner, 1):
            reg = estimator(family, settings, seed + fold)
            reg.fit(X[f], y[f])
            predictions[h] = np.asarray(reg.predict(X[h]), float)
        if not np.isfinite(predictions).all():
            raise ValueError(f'Nonfinite inner predictions: {family}')
        trial = position_mae(y, predictions, positions)
        if trial < score - 1e-10:  # fixed tie-breaking order
            best, score = dict(settings), trial
    if best is None:
        raise ValueError('No hyperparameter chosen')
    return best, float(score)


def verify_splits(folds: list[tuple[np.ndarray, np.ndarray]], positions: np.ndarray) -> None:
    p = np.asarray(positions)
    covered = np.zeros(len(p), dtype=int)
    for fit, held in folds:
        if len(fit) == 0 or len(held) == 0 or np.intersect1d(p[fit], p[held]).size:
            raise ValueError('Group leakage or empty fold')
        covered[held] += 1
    if not np.array_equal(covered, np.ones(len(p), dtype=int)):
        raise ValueError('Fold coverage is incomplete or duplicated')


def get_outer_folds(reference: pd.DataFrame, keys: np.ndarray, positions: np.ndarray):
    """Recover exact frozen Phase 15C 2x5 fold assignment, no resplitting."""
    ix = {str(k): i for i, k in enumerate(keys)}
    p = np.asarray(positions)
    all_folds = []
    for (rep, fold), held in reference.groupby(['repeat', 'fold'], sort=True):
        held_idx = np.array([ix[str(k)] for k in held.variant_key], dtype=int)
        held_pos = np.unique(p[held_idx])
        fit_idx = np.flatnonzero(~np.isin(p, held_pos))
        if np.intersect1d(p[fit_idx], held_pos).size:
            raise ValueError('Outer protein-position leakage')
        if len(fit_idx) + len(held_idx) != len(keys):
            raise ValueError('Outer fold failed to partition all training variants')
        all_folds.append((int(rep), int(fold), fit_idx, held_idx))
    if len(all_folds) != 10:
        raise ValueError('Expected exact 2x5 frozen outer folds')
    for rep in (1, 2):
        splits = [(fit, held) for r, _, fit, held in all_folds if r == rep]
        if len(splits) != 5:
            raise ValueError('Incomplete outer repeat')
        verify_splits(splits, p)
    return all_folds


def validate_phase16c(reference: pd.DataFrame, old15: pd.DataFrame, report: dict, root: Path):
    if (report.get('phase') != '16C_thermompnn_stability_comparison'
            or report.get('validation_test_outcomes_accessed') is not False
            or report.get('structure_forest_reproduced') is not True
            or report.get('n_vti_train') != 210
            or report.get('n_scored') != 195
            or report.get('phase15c_oof_sha256') != base.sha256(root / 'results/structure/phase15c_train_oof.csv')
            or report.get('stability_features_sha256') != base.sha256(root / 'results/structure/phase16_thermompnn/vti_train_stability_features.csv')
            or report.get('stability_provenance_sha256') != base.sha256(root / 'results/structure/phase16_thermompnn/thermompnn_provenance.json')):
        raise ValueError('Phase 16C provenance mismatch')
    expected = set(REFERENCES) | {f'{sb.SHUFFLE_PREFIX}{j}' for j in range(sb.N_SHUFFLES)}
    if set(reference.model) != expected:
        raise ValueError('Phase 16C reference models incomplete')
    key_cols = ['repeat', 'fold', 'variant_key']
    for model in (sb.MODEL_SEQUENCE, sb.MODEL_STRUCTURE):
        a = reference.loc[reference.model.eq(model), key_cols + ['predicted_pp', 'observed_pp']]
        b = old15.loc[old15.model.eq(model), key_cols + ['predicted_pp', 'observed_pp']]
        joined = a.merge(b, on=key_cols, suffixes=('_16c', '_15c'), validate='one_to_one', indicator=True)
        if (len(joined) != len(a) or len(joined) != len(b)
                or not joined._merge.eq('both').all()
                or not np.allclose(joined.predicted_pp_16c, joined.predicted_pp_15c, rtol=0, atol=1e-6)
                or not np.allclose(joined.observed_pp_16c, joined.observed_pp_15c, rtol=0, atol=1e-6)):
            raise ValueError(f'Phase 16C {model} differs from Phase 15C')


def summarize(oof: pd.DataFrame, n_boot: int = N_BOOT):
    if oof.duplicated(['model', 'repeat', 'variant_key']).any():
        raise ValueError('Duplicate OOF predictions')
    names = tuple(sorted(oof.model.unique()))
    baseline = set(REFERENCES)
    if not baseline.issubset(names):
        raise ValueError('Missing frozen reference models')
    for model in names:
        m = oof.loc[oof.model.eq(model)]
        n = m.variant_key.nunique()
        if n != 210 or len(m) != 2*n or not m.groupby('variant_key').repeat.nunique().eq(2).all():
            raise ValueError('Missing repeated held-out prediction: ' + model)
    avg = (oof.groupby(['model', 'variant_key', 'protein_position', 'study_cohort'], as_index=False)
           .agg(observed_pp=('observed_pp', 'mean'), predicted_pp=('predicted_pp', 'mean')))
    # Each variant has the same observed outcome across all model/repeat combinations.
    for key, group in avg.groupby('variant_key'):
        if group.observed_pp.max() - group.observed_pp.min() > 1e-8 or group.protein_position.nunique() != 1 or group.study_cohort.nunique() != 1:
            raise ValueError(f'Inconsistent OOF variant identity {key}')
    if not np.isfinite(avg[['observed_pp', 'predicted_pp']].to_numpy(float)).all():
        raise ValueError('Invalid OOF prediction')
    avg['abs_error_pp'] = (avg.observed_pp - avg.predicted_pp).abs()
    avg['signed_error_pp'] = avg.predicted_pp - avg.observed_pp
    rows, classifiers = [], []
    for cohort_idx, cohort in enumerate(('all', 'ETI, VTI', 'VTI')):
        sub = avg if cohort == 'all' else avg.loc[avg.study_cohort.eq(cohort)]
        by_pos = sub.groupby(['protein_position','model']).abs_error_pp.mean().unstack('model')
        by_bias = sub.groupby(['protein_position','model']).signed_error_pp.mean().unstack('model')
        if by_pos.isna().any().any() or by_bias.isna().any().any():
            raise ValueError('Incomplete position/model matrix')
        rng = np.random.default_rng(SEED + cohort_idx)
        draws = rng.integers(0, len(by_pos), size=(n_boot, len(by_pos)))
        for name in names:
            errs = by_pos[name].to_numpy(float)
            gain_structure = by_pos[sb.MODEL_STRUCTURE].to_numpy(float) - errs
            gain_stability = by_pos[sb.MODEL_STABILITY].to_numpy(float) - errs
            lo, hi = np.quantile(gain_structure[draws].mean(axis=1), [.025, .975])
            lo2, hi2 = np.quantile(gain_stability[draws].mean(axis=1), [.025, .975])
            rows.append(dict(cohort=cohort, model=name, n_positions=len(by_pos),
                             position_mae_pp=float(errs.mean()), mean_bias_pp=float(by_bias[name].mean()),
                             gain_vs_structural_pp=float(gain_structure.mean()),
                             gain_vs_structural_ci95=[float(lo),float(hi)],
                             gain_vs_stability_pp=float(gain_stability.mean()),
                             gain_vs_stability_ci95=[float(lo2),float(hi2)]))
            frame = sub.loc[sub.model.eq(name)].copy().sort_values('variant_key')
            nr = frame.observed_pp.to_numpy(float) < THRESHOLD
            scores = -frame.predicted_pp.to_numpy(float)
            weights = (1./frame.groupby('protein_position').variant_key.transform('size')).to_numpy(float)
            selected = np.argsort(-scores, kind='stable')[:int(np.ceil(.2*len(frame)))]
            if not nr.any() or nr.all():
                raise ValueError('Nonresponse ranking requires both assay classes in each cohort')
            classifiers.append(dict(cohort=cohort, model=name, n_variants=len(frame),
                                    n_nonresponders=int(nr.sum()),
                                    nonresponse_auroc=float(roc_auc_score(nr, scores, sample_weight=weights)),
                                    nonresponse_ap=float(average_precision_score(nr, scores, sample_weight=weights)),
                                    nr_detected_at_raw_threshold=int(np.sum(nr & (scores > -THRESHOLD))),
                                    triage_n=int(len(selected)), triage_nr=int(nr[selected].sum())))
    return pd.DataFrame(rows), pd.DataFrame(classifiers), avg


def prepare(root: Path):
    """Validate frozen manifests and load only VTI training outcomes."""
    from cftr_theratype.benchmark import make_features
    from cftr_theratype.reference import bihler_2024_assay_sequence
    from cftr_theratype.splits import parse_substitution
    from cftr_theratype.transfer import VTI_PAIRS, SPLIT_OUT, MANIFEST, RESULTS, _load_allowed_labels
    raw_paths = [VTI_PAIRS, SPLIT_OUT, MANIFEST,
                 root / 'results/structure/phase15c_train_oof.csv',
                 root / 'results/structure/phase15c_summary.json',
                 root / 'results/structure/phase15b_position_features.csv',
                 root / 'results/structure/phase15b_feature_summary.json',
                 root / 'results/structure/phase16c_stability_train_oof.csv',
                 root / 'results/structure/phase16c_stability_summary.json',
                 root / 'results/structure/phase16_thermompnn/vti_train_stability_features.csv',
                 RESULTS / 'train_cv_predictions.csv', RESULTS / 'summary.json']
    for file in raw_paths:
        if not file.is_file():
            raise FileNotFoundError(file)
    frozen = json.loads(MANIFEST.read_text())
    p15 = json.loads((root / 'results/structure/phase15c_summary.json').read_text())
    p16 = json.loads((root / 'results/structure/phase16c_stability_summary.json').read_text())
    transfer_summary = json.loads((RESULTS / 'summary.json').read_text())
    if (frozen.get('vti_source_sha256') != base.sha256(VTI_PAIRS)
            or transfer_summary.get('vti_source_sha256') != base.sha256(VTI_PAIRS)
            or transfer_summary.get('vti_partition_sha256') != base.sha256(SPLIT_OUT)
            or p15.get('phase') != '15C_structure_benchmark'
            or p15.get('baseline_reproduced_per_fold') is not True
            or p15.get('validation_or_test_outcomes_loaded') is not False
            or p15.get('vti_source_sha256') != base.sha256(VTI_PAIRS)
            or p15.get('vti_partition_sha256') != base.sha256(SPLIT_OUT)
            or p15.get('baseline_oof_sha256') != base.sha256(RESULTS / 'train_cv_predictions.csv')
            or p15.get('baseline_summary_sha256') != base.sha256(RESULTS / 'summary.json')):
        raise ValueError('Frozen original transfer and Phase 15C manifests disagree')
    # Only identities/membership (no outcomes) from partition file.
    metadata = pd.read_parquet(SPLIT_OUT, columns=['variant_key', 'protein_position',
                            'transfer_split', 'study_cohort', 'variant', 'variant_type', 'hgvs_protein'])
    if metadata.variant_key.isna().any() or metadata.variant_key.duplicated().any():
        raise ValueError('Frozen split has ambiguous identities')
    train = metadata.loc[metadata.transfer_split.eq('train')].copy()
    if len(train) != 210 or train.protein_position.nunique() != 168:
        raise ValueError('VTI TRAIN cardinality changed')
    labels = _load_allowed_labels(VTI_PAIRS, set(train.variant_key), 'vti_rescue_pp')
    train = train.merge(labels[['variant_key','vti_rescue_pp']], on='variant_key', validate='one_to_one')
    if len(train) != 210 or not np.isfinite(train.vti_rescue_pp.to_numpy(float)).all():
        raise ValueError('Invalid filtered TRAIN outcomes')
    seq = bihler_2024_assay_sequence()
    parsed = [parse_substitution(r, seq) for _,r in train.iterrows()]
    if any(p[-1] != 'valid' or p[0] != pos for p,pos in zip(parsed,train.protein_position)):
        raise ValueError('Experimental M470 reference mismatch')
    train['reference_aa'] = [p[1] for p in parsed]
    train['alternate_aa'] = [p[2] for p in parsed]
    train = train.sort_values('variant_key').reset_index(drop=True)
    old15 = pd.read_csv(root / 'results/structure/phase15c_train_oof.csv',dtype={'variant_key':str})
    sb.validate_phase15c_oof(old15, train)
    old16 = pd.read_csv(root / 'results/structure/phase16c_stability_train_oof.csv',dtype={'variant_key':str})
    validate_phase16c(old16, old15, p16, root)
    scores = sb.verify_stability_artifacts(root, set(train.variant_key))
    Xstab = sb.stability_matrix(train, scores)
    struct = base.read_feature_table(root / 'results/structure/phase15b_position_features.csv',
                                     root / 'results/structure/phase15b_feature_summary.json',root)
    Xseq, columns = make_features(train, seq)
    Xseq = np.asarray(Xseq,float)
    if Xseq.shape != (210,68) or not np.isfinite(Xseq).all():
        raise ValueError('Expected 68 finite sequence features')
    Xstruct = base.structural_matrix(train.protein_position.to_numpy(int),struct)
    feature_arrays = dict(sequence=Xseq,
                          sequence_structure=np.column_stack([Xseq,Xstruct]),
                          sequence_structure_thermompnn=np.column_stack([Xseq,Xstruct,Xstab]))
    for name, arr in feature_arrays.items():
        if np.isinf(arr).any() or len(arr) != 210:
            raise ValueError('Invalid features: '+name)
    frozen_oof = old16.loc[old16.model.isin(REFERENCES)].copy()
    if len(frozen_oof) != 3 * 2 * 210:
        raise ValueError('Frozen OOF incomplete')
    ref = old15.loc[old15.model.eq(sb.MODEL_STRUCTURE)].copy()
    outer = get_outer_folds(ref,train.variant_key.to_numpy(),train.protein_position.to_numpy())
    return train, feature_arrays, frozen_oof, outer


def run(*, smoke: bool = False):
    from cftr_theratype.transfer import ROOT, SEED as TRANSFER_SEED
    root = ROOT
    train, mats, frozen_oof, outer = prepare(root)
    y = train.vti_rescue_pp.to_numpy(float)
    positions = train.protein_position.to_numpy(int)
    print('='*98,flush=True)
    print('CFTR-THERATYPE | PHASE 17A: GROUP-NESTED TABULAR MODEL ZOO | VTI TRAIN ONLY',flush=True)
    print('='*98,flush=True)
    print('VTI TRAIN 210 variants / 168 positions | 2 x 5 frozen OUTER position folds',flush=True)
    print('INNER: 3-fold GroupKFold, position-weighted MAE; every transform fit within folds',flush=True)
    print('Features: sequence 68 | + structure 12 | + ThermoMPNN score and availability',flush=True)
    print('Models: ' + ', '.join(FAMILIES),flush=True)
    print('All data are VTI TRAIN only. Neither ETI nor VTI val/test assay outcomes are read.',flush=True)
    print('Evaluation: original frozen Forest reference predictions, 10pp raw assay NR, 20% triage',flush=True)
    if smoke:
        print('SMOKE MODE: first outer fold, only ExtraTrees and ElasticNet; no outputs written',flush=True)
        outer=outer[:1]
        families=('extra_trees','elastic_net')
        feature_names=('sequence_structure_thermompnn',)
    else:
        families=FAMILIES
        feature_names=FEATURE_SETS
    records, selections = [], []
    for rep,fold,fit,held in outer:
        # Check match original Phase 15C structural forest on each held-out fold.
        from cftr_theratype.transfer import _estimator
        Xstruct=mats['sequence_structure']
        Xfit,Xheld=base.impute_fold(Xstruct[fit,68:],Xstruct[held,68:])
        forest=_estimator('vti_forest',TRANSFER_SEED+(rep-1)*10+fold)
        forest.fit(np.column_stack([Xstruct[fit,:68],Xfit]),y[fit])
        original=forest.predict(np.column_stack([Xstruct[held,:68],Xheld]))
        ref=frozen_oof.loc[(frozen_oof.model.eq(sb.MODEL_STRUCTURE)) &
                            (frozen_oof.repeat.eq(rep)) & (frozen_oof.fold.eq(fold))]
        ref_by_key=ref.set_index('variant_key').predicted_pp
        expected=ref_by_key.loc[train.variant_key.iloc[held]].to_numpy(float)
        if not np.allclose(original,expected,rtol=0,atol=1e-6):
            raise ValueError(f'Frozen forest did not reproduce repeat={rep} fold={fold}')
        print(f'\n[OUTER] repeat={rep}/2 fold={fold}/5 train={len(fit)} held={len(held)} '
              '| original structural Forest reproduced',flush=True)
        for feat in feature_names:
            X=mats[feat]
            for family in families:
                chosen,inner_mae=inner_select(X[fit],y[fit],positions[fit],family,
                                             seed=SEED+rep*100+fold)
                reg=estimator(family,chosen,seed=SEED+rep*100+fold)
                reg.fit(X[fit],y[fit])
                pred=np.asarray(reg.predict(X[held]),dtype=float)
                if pred.shape != (len(held),) or not np.isfinite(pred).all():
                    raise ValueError(f'Invalid predictions {feat}/{family}')
                name=f'{feat}__{family}'
                selections.append(dict(repeat=rep,fold=fold,feature_set=feat,model_family=family,
                                       hyperparams=json.dumps(chosen,sort_keys=True),
                                       inner_position_mae_pp=inner_mae,n_fit=len(fit),n_held=len(held)))
                for idx,value in zip(held,pred):
                    records.append(dict(repeat=rep,fold=fold,model=name,
                                        variant_key=str(train.iloc[idx].variant_key),
                                        protein_position=int(positions[idx]),
                                        study_cohort=str(train.iloc[idx].study_cohort),
                                        observed_pp=float(y[idx]),predicted_pp=float(value)))
                print(f'  {feat:32s} {family:17s} | inner pos-MAE={inner_mae:7.3f} '
                      f'outer pos-MAE={position_mae(y[held],pred,positions[held]):7.3f} '
                      f'best={chosen}',flush=True)
    if smoke:
        print('PASS smoke mode: provenance, fold reproduction, inner CV, finite outer predictions',flush=True)
        return
    combined=pd.concat([frozen_oof,pd.DataFrame(records)],ignore_index=True)
    summary, ranking, averaged=summarize(combined)
    print('\n[RESULTS] TRAIN OOF POSITION-MAE; repeated predictions averaged by variant',flush=True)
    print('Cohort=all; lower MAE is better. Gains relative to prior structural Forest.',flush=True)
    display=summary.loc[summary.cohort.eq('all')].sort_values('position_mae_pp')
    print(f"  {'Model':51s} {'MAE':>8s} {'Gain':>9s} {'95% CI':>23s}")
    for r in display.itertuples(index=False):
        lo,hi=r.gain_vs_structural_ci95
        print(f'  {r.model:51s} {r.position_mae_pp:8.3f} {r.gain_vs_structural_pp:+9.3f} '
              f'[{lo:+9.3f},{hi:+9.3f}]')
    print('\n[NONRESPONSE] top candidate OOF ranking, training-only',flush=True)
    comparison = ranking.loc[ranking.cohort.eq('all')].merge(
        summary.loc[summary.cohort.eq('all'), ['model','position_mae_pp']],on='model')
    comparison = comparison.sort_values('position_mae_pp').head(8)
    print(comparison[['model','position_mae_pp','nonresponse_auroc','nonresponse_ap','triage_nr','triage_n']].to_string(index=False))
    print('\n[COHORTS] best 3 by overall TRAIN OOF MAE plus frozen references',flush=True)
    bests=display.model.head(3).tolist()
    print(summary.loc[summary.model.isin(set(bests) | set(REFERENCES)),
                      ['cohort','model','position_mae_pp','mean_bias_pp']].to_string(index=False))
    outdir=root/'results/model_zoo'
    outdir.mkdir(parents=True,exist_ok=True)
    paths={'oof':outdir/'phase17a_train_oof.csv',
           'selection':outdir/'phase17a_inner_selection.csv',
           'summary':outdir/'phase17a_position_mae.csv',
           'ranking':outdir/'phase17a_nonresponse.csv',
           'averaged':outdir/'phase17a_train_predictions_averaged.csv',
           'report':outdir/'phase17a_provenance.json'}
    pd.DataFrame(records).to_csv(paths['oof'],index=False)
    pd.DataFrame(selections).to_csv(paths['selection'],index=False)
    summary.to_csv(paths['summary'],index=False)
    ranking.to_csv(paths['ranking'],index=False)
    averaged.to_csv(paths['averaged'],index=False)
    import catboost, xgboost, sklearn
    files={
        'phase15c_oof':root/'results/structure/phase15c_train_oof.csv',
        'phase16c_oof':root/'results/structure/phase16c_stability_train_oof.csv',
        'phase16c_report':root/'results/structure/phase16c_stability_summary.json',
        'thermompnn_scores':root/'results/structure/phase16_thermompnn/vti_train_stability_features.csv',
    }
    report=dict(phase='17A_group_nested_model_zoo',status='exploratory_training_only',
                n_train_variants=210,n_train_positions=168,outer='original two repeats of five position-held-out folds',
                inner='three group disjoint folds, position MAE selection',
                candidate_families=list(FAMILIES),feature_sets={k:int(v.shape[1]) for k,v in mats.items()},
                grids={f:candidates(f) for f in FAMILIES},
                evaluation='variant means over repeats, then macro average of per-position error',
                label_source='VTI TRAIN only, filtered by frozen variant keys before loading outcomes',
                reference_construct='6O2P V470 E1371Q; assay M470',
                no_test_or_validation_outcomes_loaded=True,
                legacy_forests_reproduced_each_outer_fold=True,
                dependencies={'python':platform.python_version(),'sklearn':sklearn.__version__,
                              'catboost':catboost.__version__,'xgboost':xgboost.__version__,
                              'numpy':np.__version__,'pandas':pd.__version__},
                input_sha256={k:base.sha256(v) for k,v in files.items()},
                limitations=['Post-hoc exploration of previous TRAIN; best overall CV result is selection-biased',
                             'No final test, no independent validation, no causal drug-response interpretation',
                             'Raw 10 pp threshold is an in vitro assay research marker, not clinical guidance',
                             'Nested selection is inside training-position folds; final choice still requires external evaluation'],
                outputs={k:str(v.relative_to(root)) for k,v in paths.items() if k!='report'})
    paths['report'].write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('\n[OUTPUTS]',flush=True)
    for p in paths.values():
        print(' ',p.relative_to(root),flush=True)
    print('PASS: all original folds reproduced, nested group-safe tuning, no val/test outcomes loaded',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--smoke',action='store_true',help='check one outer fold and only two models; do not write outputs')
    args=ap.parse_args()
    run(smoke=args.smoke)

if __name__=='__main__':
    main()
