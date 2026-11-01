"""Phase 15D - isolate response threshold training from held-out protein positions."""
import numpy as np
import pandas as pd
import pytest
from cftr_theratype import nonresponse_classifier as mod


def test_position_weights_equalise_multiple_substitutions():
    w = mod.position_weights([10, 10, 20, 30, 30, 30])
    np.testing.assert_allclose(w, [.5, .5, 1., 1/3, 1/3, 1/3])
    assert w.sum() == pytest.approx(3)


def test_threshold_metric_tracks_nonresponders_correctly():
    t = [1, 1, 0, 0]
    proba = [.90, .30, .8, .05]
    ba, nr, responder = mod.balanced_accuracy_at_threshold(t, proba, .5, [1, 2, 3, 4])
    assert nr == pytest.approx(.5)
    assert responder == pytest.approx(.5)
    assert ba == pytest.approx(.5)


def test_inner_threshold_constrained_against_many_false_alarms():
    truth = np.array([1,1,0,0,0,0,0,0])
    scores = np.array([.8,.4,.7,.1,.1,.1,.1,.1])
    t = mod.choose_threshold_inner(truth, scores, np.arange(len(truth)),
                                   grid=np.array([.2,.4,.5,.8]),
                                   min_responder_sensitivity=.8)
    assert t == pytest.approx(.4)
    ba, nr, r = mod.balanced_accuracy_at_threshold(truth, scores, t, np.arange(len(truth)))
    assert r >= .8 and nr > 0


def test_inner_oof_grouped_same_position_never_leaks_and_finite():
    rng=np.random.default_rng(17)
    pos=np.repeat(np.arange(1,49),2)
    y=np.repeat(np.array([1]*12+[0]*36),2)
    x=rng.standard_normal((len(pos),3)); x[:,0] += y*.5
    z=rng.standard_normal((len(pos),2)); z[::13,0]=np.nan
    p=mod.inner_oof_probability(x,z,y,pos,use_structure=True,seed=1)
    assert len(p)==len(y) and np.isfinite(p).all()
    assert np.all((p>=0)&(p<=1))


def test_train_fold_needs_both_classes():
    with pytest.raises(ValueError, match='Insufficient independent'):
        mod.inner_oof_probability(np.ones((9,3)),np.ones((9,2)),[0]*8+[1],np.arange(9),
                                  use_structure=False,seed=1)


def toy_oof():
    values = [2., 5., 8., 9., 10., 15., 17., 20., 4., 12., 14., 16.]
    labels = ["ETI, VTI"]*6 + ["VTI"]*6
    rows=[]
    for rep in (1,2):
        for i,(value,cohort) in enumerate(zip(values, labels)):
            for name in mod.ALL_MODELS:
                if name == 'always_responder':
                    score,margin=0.,-1.
                elif name in ('sequence_regression','structure_regression'):
                    score,margin=-(value+2),10-(value+2)
                else:
                    score,margin=(.7 if value<10 else .3),(.2 if value<10 else -.2)
                rows.append({'repeat':rep,'model':name,'variant_key':f'v{i}',
                             'protein_position':i+1,'study_cohort':cohort,
                             'observed_pp':value, 'nr_rank_score':score,
                             'nr_decision_margin':margin})
    return pd.DataFrame(rows)


def test_two_repeat_average_and_cohort_metrics():
    avg=mod.average_repeats(toy_oof())
    assert len(avg)==len(mod.ALL_MODELS)*12
    score=mod.summarise_threshold(avg)
    assert len(score)==len(mod.ALL_MODELS)*3
    direct=score.loc[score.cohort.eq('all')&score.model.eq('sequence_balanced_fixed')].iloc[0]
    assert direct.nonresponders_detected==5
    assert direct.false_nonresponse_alarms==0
    assert direct.balanced_accuracy==pytest.approx(1.)
    always=score.loc[score.cohort.eq('all')&score.model.eq('always_responder')].iloc[0]
    assert always.nr_recall==0
    assert always.responder_recall==1
    assert always.balanced_accuracy==pytest.approx(.5)


def test_repeat_incomplete_or_metadata_disagreement_fails():
    base=toy_oof()
    with pytest.raises(ValueError,match='Missing repeated'):
        mod.average_repeats(base.iloc[:-1])
    modified=base.copy()
    modified.loc[0,'study_cohort']='wrong'
    with pytest.raises(ValueError,match='Inconsistent'):
        mod.average_repeats(modified)


def test_probability_positive_class_is_nonresponder():
    rng=np.random.default_rng(4)
    a=rng.standard_normal((80,5))
    y=(a[:,0]>1).astype(int)
    m=mod.make_classifier(13)
    m.fit(a,y)
    p=mod.predict_nonresponse_proba(m,a)
    assert len(p)==80 and np.isfinite(p).all()
    assert p[y==1].mean()>p[y==0].mean()
