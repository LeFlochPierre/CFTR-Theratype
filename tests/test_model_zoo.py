"""Phase 17A model zoo: fold safety, inner hyperparameters, OOF metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cftr_theratype import model_zoo as mz


def test_six_families_and_three_predeclared_feature_sets():
    assert len(mz.FAMILIES)==6
    assert len(mz.FEATURE_SETS)==3
    assert sum(len(mz.candidates(f)) for f in mz.FAMILIES)==17
    for family in mz.FAMILIES:
        assert len(mz.candidates(family))>=2
    with pytest.raises(ValueError):
        mz.candidates('not_a_model')


def test_exact_protein_position_macro_mae():
    assert mz.position_mae([0,0,0,0], [10,10,10,4], [1,1,1,2])==pytest.approx(7.0)
    with pytest.raises(ValueError):
        mz.position_mae([1],[np.nan],[5])


def test_split_validation_catches_leakage_and_partial_folds():
    pos=np.array([1,1,2,3,4,5])
    mz.verify_splits([(np.array([2,3,4,5]),np.array([0,1])),
                      (np.array([0,1,4,5]),np.array([2,3])),
                      (np.array([0,1,2,3]),np.array([4,5]))],pos)
    with pytest.raises(ValueError,match='leakage|empty'):
        mz.verify_splits([(np.array([0,2,3,4,5]),np.array([1]))],pos)
    with pytest.raises(ValueError,match='incomplete'):
        mz.verify_splits([(np.array([2,3,4,5]),np.array([0,1]))],pos)


def test_original_outer_folds_are_reconstructed_not_resplit():
    positions=np.array([11,11,12,13,14,14,15,16,17,18])
    keys=np.array([f'v{i}' for i in range(len(positions))])
    rows=[]
    for rep in (1,2):
        pos_folds={p: i%5+1 for i,p in enumerate(np.unique(positions))}
        if rep==2:
            pos_folds={p: (i+2)%5+1 for i,p in enumerate(np.unique(positions))}
        for v,p in zip(keys,positions):
            rows.append(dict(repeat=rep,fold=pos_folds[p],variant_key=v))
    folds=mz.get_outer_folds(pd.DataFrame(rows),keys,positions)
    assert len(folds)==10
    for rep,fold,fit,held in folds:
        assert not set(positions[fit]) & set(positions[held])
        assert len(fit)+len(held)==len(keys)
    bad=pd.DataFrame(rows).iloc[:-1]
    with pytest.raises(ValueError):
        mz.get_outer_folds(bad,keys,positions)


def test_inner_search_group_safe_and_imputation():
    rng=np.random.default_rng(17)
    positions=np.repeat(np.arange(1,13),2)
    X=rng.normal(size=(24,4))
    X[::4,2]=np.nan
    y=2*X[:,0]+rng.normal(size=24)
    best, score=mz.inner_select(X,y,positions,'elastic_net',seed=5)
    assert best in mz.candidates('elastic_net')
    assert np.isfinite(score)
    model=mz.estimator('elastic_net',best,seed=5)
    model.fit(X,y)
    assert np.isfinite(model.predict(X)).all()
    with pytest.raises(ValueError,match='Too few'):
        mz.inner_select(X[:4],y[:4],positions[:4],'elastic_net',5)


@pytest.mark.parametrize('family',mz.FAMILIES)
def test_each_estimator_smoke(family):
    rng=np.random.default_rng(19)
    X=rng.normal(size=(25,5))
    X[0,3]=np.nan
    y=10+X[:,0]*5+rng.normal(size=25)
    model=mz.estimator(family,mz.candidates(family)[0],seed=5)
    model.fit(X,y)
    assert np.isfinite(model.predict(X)).all()


def toy_oof(n=210):
    rows=[]
    names=(*mz.REFERENCES,'sequence__elastic_net')
    for rep in (1,2):
        for i in range(n):
            true=5.0 if i%5==0 else 25.0
            for name in names:
                error={mz.REFERENCES[0]:8.,mz.REFERENCES[1]:5.,
                       mz.REFERENCES[2]:4.,'sequence__elastic_net':3.}[name]
                rows.append(dict(repeat=rep,fold=i%5+1,model=name,variant_key=f'v{i}',
                                 protein_position=i+1,
                                 study_cohort='ETI, VTI' if i%2==0 else 'VTI',
                                 observed_pp=true,predicted_pp=true+error))
    return pd.DataFrame(rows)


def test_equal_position_bootstrap_and_ranking():
    scores, nr, avg=mz.summarize(toy_oof(),n_boot=70)
    row=scores.loc[scores.cohort.eq('all') & scores.model.eq('sequence__elastic_net')].iloc[0]
    assert row.position_mae_pp==pytest.approx(3.)
    assert row.gain_vs_structural_pp==pytest.approx(2.)
    assert row.gain_vs_stability_pp==pytest.approx(1.)
    assert row.gain_vs_structural_ci95==pytest.approx([2.,2.])
    assert len(avg)==4*210
    sub=nr.loc[nr.cohort.eq('all') & nr.model.eq('sequence__elastic_net')].iloc[0]
    assert sub.triage_n==42
    assert sub.n_nonresponders==42


def test_incomplete_out_of_fold_rejected():
    oof=toy_oof()
    with pytest.raises(ValueError,match='Missing repeated'):
        mz.summarize(oof.iloc[:-1])


def test_reference_model_order_contract():
    assert mz.REFERENCES==('vti_forest','structural_forest','structural_plus_thermompnn')
