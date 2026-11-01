#!/usr/bin/env python3
"""Phase 27E-1 | Label-regime mixture-of-experts audit (VTI TRAIN only).

Compare pre-specified hard and soft classification-gated regression with a
frozen global regression baseline on the ORIGINAL two-repeat, five-fold,
protein-position-disjoint outer splits. The gates are precomputed OOF scores
from Phase 27B-3; no held-out labels are used to fit a gate or an expert.

The ORACLE uses held-out labels solely for a diagnostic and MUST NOT be used as
any deployable model result. No VTI VAL / TEST outcome is read.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results/biology/phase27e"
REF = ROOT / "results/biology/phase27d/phase27d4b_regression_oof.csv"
GATES = ROOT / "results/biology/phase27b/phase27b3_repeated_oof.csv"
KEY = ["repeat", "fold", "variant_key", "protein_position"]
GATENAMES = ["coupled_binary", "multinomial_logistic"]
CLASSES = ["low", "middle", "high"]
ALPHA = 30.0
SHRINK = 0.5


def classes(y):
    y = np.asarray(y, float)
    return np.where(y < 10.0, 0, np.where(y > 40.0, 2, 1)).astype(int)


def check_gate(p):
    p = np.asarray(p, dtype=float)
    if (p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all()
            or (p < -1e-8).any() or (p > 1 + 1e-8).any()
            or not np.allclose(p.sum(axis=1), 1, atol=1e-6)):
        raise ValueError("Invalid categorical gate: probability rows must sum to one")
    return np.maximum(0.0, p) / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)


def experts_on_fold(X, y, tr, te, seed=27):
    """Three regularized experts, each fitted ONLY on its outer-train regime.

    Features are the original finite sequence descriptors (68). Train
    means and bounded predictions respect the regime in *corrector* pp.
    Training / inference code does not inspect y[te].
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    tr, te = np.asarray(tr, int), np.asarray(te, int)
    cats = classes(y[tr])
    preds = np.empty((len(te), 3), float)
    means = np.empty(3, float)
    counts = np.bincount(cats, minlength=3)
    if np.any(counts < 4):
        raise ValueError(f"Not enough TRAIN variants in an expert: {counts.tolist()}")
    for c in range(3):
        ix = tr[cats == c]
        means[c] = float(np.mean(y[ix]))
        model = make_pipeline(StandardScaler(), Ridge(alpha=ALPHA))
        model.fit(X[ix, :68], y[ix])
        preds[:, c] = model.predict(X[te, :68])
    # Enforce corrector response regimes only for individual experts.
    preds[:, 0] = np.minimum(preds[:, 0], 10.0)
    preds[:, 1] = np.clip(preds[:, 1], 10.0, 40.0)
    preds[:, 2] = np.maximum(preds[:, 2], 40.0)
    if not np.isfinite(preds).all() or not np.isfinite(means).all():
        raise ValueError("Nonfinite expert output")
    return preds, means, counts


def predict_variants(global_prediction, expert_prediction, expert_means, gate):
    """Predicted methods only; never use ground-truth held-out categories."""
    gate = check_gate(gate)
    x = np.asarray(global_prediction, float)
    e = np.asarray(expert_prediction, float)
    means = np.asarray(expert_means, float)
    if e.shape != gate.shape or x.shape != (len(gate),) or means.shape != (3,):
        raise ValueError("Global, expert and gate dimensions disagree")
    hard = e[np.arange(len(e)), gate.argmax(axis=1)]
    soft = (gate * e).sum(axis=1)
    soft_mean = gate @ means
    return {"hard": hard, "soft": soft, "soft50": SHRINK * x + (1-SHRINK) * soft,
            "soft_mean": soft_mean}


def load_saved(X, keys, positions, corrector, total, folds):
    if not REF.is_file() or not GATES.is_file():
        raise FileNotFoundError(f"Run Phase 27D-4B and 27B-3 first: {REF}, {GATES}")
    ref = pd.read_csv(REF, dtype={"variant_key": str})
    gr = pd.read_csv(GATES, dtype={"variant_key": str})
    required_ref = {"corrector_frozen", "total_frozen", "observed_corrector", "observed_total", *KEY}
    required_gate = {"model", "true_class", "p_low", "p_middle", "p_high", *KEY}
    if not required_ref.issubset(ref) or not required_gate.issubset(gr):
        raise ValueError(f"Missing reference columns: {sorted(required_ref-set(ref))}; gates: {sorted(required_gate-set(gr))}")
    if len(ref) != 2*len(keys) or ref.duplicated(KEY).any():
        raise ValueError("Frozen regression OOF is incomplete or duplicated")
    rr = ref.set_index(KEY, verify_integrity=True)
    gates = {}
    for name in GATENAMES:
        part = gr.loc[gr.model.eq(name)].copy()
        if len(part) != len(ref) or part.duplicated(KEY).any():
            raise ValueError(f"Bad gate membership: {name}")
        if set(pd.MultiIndex.from_frame(part[KEY])) != set(pd.MultiIndex.from_frame(ref[KEY])):
            raise ValueError(f"Different frozen outer-fold membership: {name}")
        part = part.set_index(KEY, verify_integrity=True)
        check_gate(part[["p_low", "p_middle", "p_high"]].to_numpy(float))
        gates[name] = part
    for rep, fold, train_keys, test_keys in folds:
        te = np.flatnonzero(np.isin(keys, list(test_keys)))
        tr = np.flatnonzero(np.isin(keys, list(train_keys)))
        if len(te)+len(tr) != len(keys) or set(positions[te]) & set(positions[tr]):
            raise ValueError("Outer fold leakage / incomplete membership")
        idx = [(int(rep), int(fold), keys[i], int(positions[i])) for i in te]
        s = rr.loc[idx]
        if not np.allclose(s.observed_corrector.to_numpy(float), corrector[te], atol=1e-6):
            raise ValueError("Frozen corrector outcome mismatch")
        if not np.allclose(s.observed_total.to_numpy(float), total[te], atol=1e-6):
            raise ValueError("Frozen total outcome mismatch")
        for name, frame in gates.items():
            if not np.array_equal(frame.loc[idx].true_class.to_numpy(int), classes(corrector[te])):
                raise ValueError(f"Frozen categorical label mismatch: {name}")
    return rr, gates


def calculate_oof(X, y, yt, keys, positions, folds, reference, saved_gates):
    """Fits all regime experts only on the outer-training partition."""
    records = []
    fold_qc = []
    for fi, (rep, fold, train_keys, test_keys) in enumerate(folds, 1):
        tr = np.flatnonzero(np.isin(keys, list(train_keys)))
        te = np.flatnonzero(np.isin(keys, list(test_keys)))
        if (len(tr) + len(te) != len(keys) or
                len(set(positions[tr]) & set(positions[te])) != 0):
            raise ValueError(f"Train/test leakage in fold {rep}:{fold}")
        idx = [(int(rep), int(fold), keys[i], int(positions[i])) for i in te]
        frozen = reference.loc[idx]
        global_c = frozen.corrector_frozen.to_numpy(float)
        frozen_acute = (frozen.total_frozen - frozen.corrector_frozen).to_numpy(float)
        e, means, counts = experts_on_fold(X, y, tr, te, fi+2700)
        # y[te] accessed ONLY for oracle diagnostic and final scoring, never expert fitting.
        oracle_ridge = e[np.arange(len(te)), classes(y[te])]
        oracle_mean = means[classes(y[te])]
        candidates = {"global": global_c, "oracle_ridge": oracle_ridge,
                      "oracle_mean": oracle_mean}
        for name, frame in saved_gates.items():
            gate = frame.loc[idx, ["p_low", "p_middle", "p_high"]].to_numpy(float)
            preds = predict_variants(global_c, e, means, gate)
            for suffix, values in preds.items():
                candidates[f"{suffix}_{name}"] = values
        for j, i in enumerate(te):
            row = {"repeat": int(rep), "fold": int(fold),
                   "variant_key": str(keys[i]), "protein_position": int(positions[i]),
                   "observed_corrector": float(y[i]), "observed_total": float(yt[i]),
                   "expert_low": float(e[j,0]), "expert_middle": float(e[j,1]),
                   "expert_high": float(e[j,2])}
            for name, values in candidates.items():
                row["corrector_" + name] = float(values[j])
                row["total_" + name] = float(values[j] + frozen_acute[j])
            records.append(row)
        fold_qc.append({"repeat":int(rep), "fold":int(fold),
                        "train_variants":len(tr), "test_variants":len(te),
                        "train_positions":len(np.unique(positions[tr])),
                        "test_positions":len(np.unique(positions[te])),
                        "train_low":int(counts[0]), "train_middle":int(counts[1]),
                        "train_high":int(counts[2]),
                        "mean_low":float(means[0]), "mean_middle":float(means[1]),
                        "mean_high":float(means[2]),
                        "oracle_uses_test_labels": True})
        print(f"completed {fi}/{len(folds)}: rep={rep} fold={fold} "
              f"train classes={counts.tolist()} held positions={len(np.unique(positions[te]))}", flush=True)
    d = pd.DataFrame(records)
    if len(d) != 2*len(keys) or d.duplicated(KEY).any():
        raise ValueError("Incomplete repeated OOF predictions")
    if not np.isfinite(d.select_dtypes(include=["number"]).to_numpy()).all():
        raise ValueError("Nonfinite outputs")
    return d, pd.DataFrame(fold_qc)


def position_mae(d, actual, prediction):
    z = d[["repeat", "protein_position", actual, prediction]].copy()
    z["error"] = (z[prediction] - z[actual]).abs()
    return float(z.groupby(["repeat", "protein_position"]).error.mean()
                 .groupby("protein_position").mean().mean())


def bootstrap_gain(d, actual, candidate, base, nboot=2000, seed=27):
    z = d[["repeat", "protein_position", actual, candidate, base]].copy()
    z["gain"] = (z[base]-z[actual]).abs() - (z[candidate]-z[actual]).abs()
    values = (z.groupby(["repeat", "protein_position"]).gain.mean()
              .groupby("protein_position").mean().to_numpy(float))
    rng = np.random.default_rng(seed)
    ii = rng.integers(len(values), size=(nboot, len(values)))
    samples = values[ii].mean(axis=1)
    return {"mae_gain_pp":float(values.mean()),
            "ci95_low":float(np.quantile(samples,.025)),
            "ci95_high":float(np.quantile(samples,.975)),
            "positions":int(len(values))}


def report(d, nboot):
    methods = [c.removeprefix("corrector_") for c in d.columns
               if c.startswith("corrector_")]
    overview, subgroup = [], []
    for outcome in ["corrector", "total"]:
        actual = "observed_" + outcome
        y = d[actual].to_numpy(float)
        for method in methods:
            pred = outcome + "_" + method
            overview.append({"target":outcome, "model":method,
                             "position_mae_pp":position_mae(d, actual, pred),
                             "variant_mae_pp":float(np.mean(np.abs(d[pred]-d[actual]))),
                             "signed_bias_pp":float(np.mean(d[pred]-d[actual])),
                             "rmse_pp":float(np.sqrt(np.mean((d[pred]-d[actual])**2)))})
            for label, mask in [("low<10",y<10),("middle10-40",(y>=10)&(y<=40)),("high>40",y>40)]:
                q=d.loc[mask]
                subgroup.append({"target":outcome, "model":method, "subset":label,
                                 "variants":q.variant_key.nunique(),
                                 "positions":q.protein_position.nunique(),
                                 "position_mae_pp":position_mae(q,actual,pred),
                                 "signed_bias_pp":float(np.mean(q[pred]-q[actual]))})
    ov = pd.DataFrame(overview)
    sb = pd.DataFrame(subgroup)
    comparisons = ["hard_coupled_binary", "soft_coupled_binary", "soft50_coupled_binary",
                   "hard_multinomial_logistic", "soft_multinomial_logistic", "soft50_multinomial_logistic",
                   "soft_mean_coupled_binary", "soft_mean_multinomial_logistic",
                   "oracle_mean", "oracle_ridge"]
    boot = []
    for i, target in enumerate(["corrector", "total"]):
        for j, method in enumerate(comparisons):
            boot.append({"target":target,"model":method,"baseline":"global",
                         **bootstrap_gain(d, "observed_"+target, target+"_"+method,
                                          target+"_global",nboot, seed=27000+i*100+j)})
    return ov, sb, pd.DataFrame(boot)


def self_test():
    rng = np.random.default_rng(27)
    n = 90
    pos = np.repeat(np.arange(45),2)
    X = rng.normal(size=(n,80))
    X[:,68:]=np.nan
    truth = np.where(np.arange(n)%3==0,5.,np.where(np.arange(n)%3==1,25.,60.))
    truth = truth + rng.normal(0,2,n)
    yt = truth + rng.normal(6,2,n)
    keys = np.array([f"v{i}" for i in range(n)])
    from sklearn.model_selection import GroupKFold
    folds=[]
    ref=[]
    gates={name:[] for name in GATENAMES}
    for rep in (1,2):
        # GroupKFold with shuffled block identities across the two repeats.
        groups = pos if rep==1 else (pos*11+3)%47
        splits = GroupKFold(n_splits=5).split(X,groups=groups)
        for fi,(tr,te) in enumerate(splits,1):
            # In synthetic data both grouping schemes must respect canonical positions.
            if set(pos[tr]) & set(pos[te]):
                raise AssertionError('Synthetic grouping failed')
            for i in te:
                item=dict(repeat=rep,fold=fi,variant_key=keys[i],protein_position=int(pos[i]),
                          corrector_frozen=float(truth[i]+5),total_frozen=float(truth[i]+12),
                          observed_corrector=float(truth[i]),observed_total=float(yt[i]))
                ref.append(item)
                for name in GATENAMES:
                    p=np.array([.2,.6,.2])
                    if name=="multinomial_logistic":p=np.array([.3,.3,.4])
                    gates[name].append({"repeat":rep,"fold":fi,"variant_key":keys[i],
                                        "protein_position":int(pos[i]),"p_low":p[0],
                                        "p_middle":p[1],"p_high":p[2]})
            folds.append((rep,fi,set(keys[tr]),set(keys[te])))
    rr=pd.DataFrame(ref).set_index(KEY)
    gg={name:pd.DataFrame(gates[name]).set_index(KEY) for name in GATENAMES}
    d,qc=calculate_oof(X,truth,yt,keys,pos,folds,rr,gg)
    assert len(d)==2*n and len(qc)==10
    assert np.all(d.expert_low<=10) and np.all((d.expert_middle>=10)&(d.expert_middle<=40))
    assert np.all(d.expert_high>=40)
    assert np.allclose(d.corrector_soft50_coupled_binary,
                       .5*d.corrector_global+.5*d.corrector_soft_coupled_binary)
    assert np.isclose(position_mae(d,"observed_corrector","corrector_global"),5,atol=1e-10)
    a,b,c=report(d,200)
    assert len(a)>=10 and len(b)>0 and len(c)==20
    assert np.isfinite(c.mae_gain_pp.to_numpy()).all()
    print("PASS synthetic: 2x5 position-disjoint expert fits, hard/soft/oracle routes, total reconstruction, bootstrap")


def main(nboot):
    import phase26h_auxiliary_corrector as h
    v,X,_,_,folds=h.load_data()
    X=np.asarray(X,float)
    keys=v.variant_key.astype(str).to_numpy()
    pp=v.protein_position.to_numpy(int)
    yc=v.corrector_rescue_pp.to_numpy(float)
    yt=v.vti_rescue_pp.to_numpy(float)
    if (X.shape!=(210,80) or len(set(keys))!=210 or len(np.unique(pp))!=168
            or len(folds)!=10 or not np.isfinite(X[:,:68]).all()
            or not np.isfinite(yc).all() or not np.isfinite(yt).all()
            or list(np.bincount(classes(yc), minlength=3)) != [52,129,29]):
        raise ValueError("Unexpected frozen VTI TRAIN cohort/features/labels")
    ref,gates=load_saved(X,keys,pp,yc,yt,folds)
    print("="*108)
    print("PHASE 27E-1 | CLASSIFICATION-GATED LOW/MIDDLE/HIGH RESCUE EXPERTS | TRAIN ONLY")
    print("="*108)
    print("210 variants / 168 positions; original frozen 2x5 grouped folds; 52 low / 129 middle / 29 high")
    print("Frozen gates: coupled_binary and multinomial_logistic (27B3) — scores not calibrated")
    print("Expert head: fold-local StandardScaler + Ridge(alpha=30), sequence68, trained on category-specific TRAIN labels")
    print("Hard, soft, 50%-shrunk soft, gate+class-means. ORACLE uses HELD-OUT LABELS — DIAGNOSTIC ONLY.")
    print("Corrector predictions reconstitute total VTI with unchanged frozen acute component")
    print("No VAL / TEST outcome access; no architecture/tuning based on previously opened test")
    d,qc=calculate_oof(X,yc,yt,keys,pp,folds,ref,gates)
    overview,subgroups,boot=report(d,nboot)
    OUT.mkdir(parents=True,exist_ok=True)
    d.to_csv(OUT/"phase27e1_repeated_oof.csv",index=False)
    qc.to_csv(OUT/"phase27e1_fold_qc.csv",index=False)
    overview.to_csv(OUT/"phase27e1_metrics.csv",index=False)
    subgroups.to_csv(OUT/"phase27e1_subgroups.csv",index=False)
    boot.to_csv(OUT/"phase27e1_bootstrap.csv",index=False)
    print("\nA. TRAINING SUBGROUPS PER FOLD")
    print(qc[['repeat','fold','train_low','train_middle','train_high','mean_low','mean_middle','mean_high']].round(3).to_string(index=False))
    print("\nB. OVERALL POSITION-WEIGHTED MAE (pp) / SIGNED BIAS")
    print(overview.round(4).to_string(index=False))
    print("\nC. CORRECTOR MAE BY TRUE RESPONSE REGIME")
    print(subgroups.loc[subgroups.target.eq('corrector'), ['model','subset','variants','positions','position_mae_pp','signed_bias_pp']].round(4).to_string(index=False))
    print("\nD. PAIRED POSITION-CLUSTER BOOTSTRAP: POSITIVE GAIN = LOWER MAE")
    print(boot.round(4).to_string(index=False))
    print("\nE. GATE CONFUSION MATRICES (averaged over repeat predictions)")
    y_av = d.groupby('variant_key').observed_corrector.first()
    for name,frame in gates.items():
        g = frame.reset_index().groupby('variant_key')[["p_low","p_middle","p_high"]].mean()
        if set(g.index)!=set(y_av.index):raise ValueError("Gate averaged membership mismatch")
        ytrue=classes(y_av.loc[g.index].to_numpy(float))
        pred=g.to_numpy(float).argmax(axis=1)
        print(name, 'rows actual low/mid/high; cols predicted low/mid/high')
        print(confusion_matrix(ytrue,pred,labels=[0,1,2]))
    print("\nINTERPRETATION: Oracle results reveal the value of correct routing, not achievable prospective performance.")
    print("Hard gating can create large errors from wrong routing. Soft gating may still regress to the mean.")
    print("All differences are exploratory on extensively reused TRAIN folds; avoid picking a winner from this grid.")
    print("PASS Phase 27E-1 completed; VAL and TEST unopened. Outputs:",OUT)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--bootstraps',type=int,default=2000)
    a=parser.parse_args()
    if a.self_test:self_test()
    else:
        if a.bootstraps<100:parser.error('--bootstraps must be >=100')
        main(a.bootstraps)
