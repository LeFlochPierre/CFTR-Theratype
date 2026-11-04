#!/usr/bin/env python3
"""Phase 32A: adaptive, cost-aware VTI experimental selection simulation.

Only the frozen 210 VTI TRAIN variants are read. An Oracle hides unmeasured
VTI and baseline outcomes from acquisition policies. Protein positions are
acquisition units and outer-fold grouping units.

All assay costs are *hypothetical units*, NOT validated laboratory costs.
Cheap baseline screens are retrospective counterfactual actions: feasibility
of physically separating baseline and VTI assays is NOT established.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results/biology/phase32a"
TARGETS = {"total": "vti_rescue_pp", "corrector": "corrector_rescue_pp"}


def get_real():
    from cftr_theratype.datasets.frozen_vti import load_train
    d, features, _, folds, source = load_train(ROOT)
    x = np.asarray(features["sequence_structure"], float)
    if len(d) != 210 or d.protein_position.nunique() != 168 or x.shape != (210, 80):
        raise ValueError("Frozen 210/168/80 dataset mismatch")
    if not d.transfer_split.eq("train").all() or len(folds) != 10:
        raise ValueError("Incorrect frozen partition / outer folds")
    if not np.isfinite(d[["baseline_fsk_pct_wt", *TARGETS.values()]].to_numpy(float)).all():
        raise ValueError("TRAIN response values missing")
    esm = None
    scorepath = ROOT / "results/biology/phase31m/esmrank_output/phase31m_esmrank_input/predictions.csv"
    if scorepath.exists():
        from phase31m_three_tracks import get_data, load_esm
        aligned, _, _, meta = get_data()
        if not aligned.variant_key.astype(str).equals(d.variant_key.astype(str)):
            raise ValueError("ESMRank VTI TRAIN order mismatch")
        scores = load_esm(scorepath)
        if not set(meta.aa31i).issubset(scores.index):
            raise ValueError("ESMRank score missing frozen VTI TRAIN substitutions")
        esm = meta.aa31i.map(scores).to_numpy(float)
        if not np.isfinite(esm).all():
            raise ValueError("Nonfinite ESMRank scores")
    return d, x, folds, esm, source


class AssayOracle:
    """Policies can buy assays; hidden labels never enter policy-facing arrays."""
    def __init__(self, pos, y, baseline, allowed):
        self._pos = np.asarray(pos, int)
        self._y = np.asarray(y, float)
        self._baseline = np.asarray(baseline, float)
        self.allowed = set(map(int, allowed))
        self.vti = {}  # position -> (row indices, true VTI labels)
        self.screen = {}  # position -> mean baseline measurement

    def _rows(self, position):
        position = int(position)
        if position not in self.allowed:
            raise AssertionError("LEAKAGE: attempted to purchase a held-out assay")
        return np.flatnonzero(self._pos == position)

    def buy_vti(self, position):
        position = int(position)
        if position in self.vti:
            raise AssertionError("Duplicate VTI assay")
        rows = self._rows(position)
        self.vti[position] = (rows, self._y[rows].copy())
        # The stored VTI measurement already includes the untreated baseline.
        self.screen[position] = float(np.mean(self._baseline[rows]))

    def buy_baseline(self, position):
        position = int(position)
        if position in self.screen:
            raise AssertionError("Duplicate baseline purchase")
        rows = self._rows(position)
        self.screen[position] = float(np.mean(self._baseline[rows]))

    def labelled(self):
        entries = list(self.vti.values())
        indices = np.concatenate([e[0] for e in entries])
        values = np.concatenate([e[1] for e in entries])
        return indices, values


class Problem:
    def __init__(self, d, x, trainkeys, testkeys, target, esm):
        self.keys = d.variant_key.astype(str).to_numpy()
        self.pos = d.protein_position.to_numpy(int)
        self.x = np.asarray(x, float)
        self.y = d[TARGETS[target]].to_numpy(float)
        self.base = d.baseline_fsk_pct_wt.to_numpy(float)
        self.train_rows = np.flatnonzero(np.isin(self.keys, list(trainkeys)))
        self.test_rows = np.flatnonzero(np.isin(self.keys, list(testkeys)))
        if len(self.train_rows) + len(self.test_rows) != len(d):
            raise AssertionError("Outer fold membership incomplete")
        self.train_pos = np.unique(self.pos[self.train_rows])
        self.test_pos = np.unique(self.pos[self.test_rows])
        if set(self.train_pos) & set(self.test_pos):
            raise AssertionError("Outer-fold position leakage")
        self.prow = {int(p): np.flatnonzero(self.pos == p) for p in self.train_pos}
        # Mutation-only features, averaged per candidate position.
        all_features = np.array([pd.DataFrame(self.x[self.prow[int(p)]]).mean(axis=0).to_numpy(float)
                                 for p in self.train_pos])
        # Impute and scale using genotype features from candidate outer TRAIN only.
        imp = SimpleImputer(strategy="median", keep_empty_features=True)
        scaled = StandardScaler().fit_transform(imp.fit_transform(all_features))
        self.feature = {int(p): scaled[i] for i, p in enumerate(self.train_pos)}
        self.esm = None
        if esm is not None:
            esm = np.asarray(esm, float)
            self.esm = {int(p): float(np.mean(esm[self.prow[int(p)]])) for p in self.train_pos}

    def new_oracle(self):
        return AssayOracle(self.pos, self.y, self.base, self.train_pos)

    def model(self, oracle):
        from cftr_theratype.models.regressors import phase26g_model
        idx, values = oracle.labelled()
        model = phase26g_model("xgb", (2, 5.0, 180))
        model.fit(self.x[idx], values)
        return model

    def median(self, oracle):
        idx, y = oracle.labelled()
        return float(pd.DataFrame({"p": self.pos[idx], "y": y}).groupby("p").y.mean().median())

    def remaining(self, oracle):
        return sorted(set(map(int, self.train_pos)) - set(oracle.vti))


def diversity(pool, chosen, features, n):
    selected = []
    anchor = list(chosen)
    for _ in range(min(n, len(pool))):
        remaining = [p for p in pool if p not in selected]
        if not remaining:
            break
        if not anchor:
            pick = remaining[0]
        else:
            distances = [min(float(np.linalg.norm(features[p] - features[q]))
                             for q in anchor) for p in remaining]
            pick = remaining[int(np.argmax(distances))]
        selected.append(pick)
        anchor.append(pick)
    return selected


def choose_direct(policy, problem, oracle, rng, batch):
    candidates = problem.remaining(oracle)
    if len(candidates) < batch:
        raise RuntimeError("Not enough candidate positions for requested budget")
    if policy == "random":
        return list(map(int, rng.choice(candidates, size=batch, replace=False)))
    if policy == "diversity":
        return diversity(candidates, list(oracle.vti), problem.feature, batch)
    if policy == "esmrank_diversity":
        # Genotype-only scalar; choose additions far from already measured score values.
        if problem.esm is None:
            raise ValueError("ESMRank scores unavailable")
        out, selected = [], list(oracle.vti)
        for _ in range(batch):
            other = [p for p in candidates if p not in out]
            distance = [min(abs(problem.esm[p] - problem.esm[q]) for q in selected)
                        for p in other]
            p = other[int(np.argmax(distance))]
            out.append(p)
            selected.append(p)
        return out
    fitted = problem.model(oracle)
    pospred = np.array([fitted.predict(problem.x[problem.prow[p]]).mean()
                        for p in candidates])
    if policy == "model_high":
        return [candidates[i] for i in np.argsort(-pospred, kind="stable")[:batch]]
    if policy == "uncertainty":
        idx, y = oracle.labelled()
        forest = ExtraTreesRegressor(n_estimators=60, min_samples_leaf=2,
                                     max_features=0.8, random_state=3917, n_jobs=3)
        # Impute using ONLY revealed VTI TRAIN variants.
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        xa = imputer.fit_transform(problem.x[idx])
        forest.fit(xa, y)
        xc = imputer.transform(np.vstack([
            pd.DataFrame(problem.x[problem.prow[p]]).mean(axis=0).to_numpy(float) for p in candidates]))
        ensemble = np.stack([tree.predict(xc) for tree in forest.estimators_])
        uncertainty = ensemble.std(axis=0)
        return [candidates[i] for i in np.argsort(-uncertainty, kind="stable")[:batch]]
    raise ValueError(policy)


def choose_screen(problem, oracle, batch, confirms):
    # Spend batch cheap screens on unseen candidate positions, using diversity.
    unseen = [p for p in problem.remaining(oracle) if p not in oracle.screen]
    selected = diversity(unseen, list(oracle.screen), problem.feature, batch)
    if len(selected) != batch:
        raise RuntimeError("Not enough unscreened candidates")
    for p in selected:
        oracle.buy_baseline(p)
    # Learn the relationship from previously *revealed VTI positions only*.
    # No baseline or VTI outcome from outer held-out positions is available.
    bought = sorted(oracle.vti)
    xx = np.array([[oracle.screen[p]] for p in bought])
    yy = np.array([oracle.vti[p][1].mean() for p in bought])
    scale = StandardScaler().fit(xx)
    model = Ridge(alpha=10.0).fit(scale.transform(xx), yy)
    candidates = [p for p in problem.remaining(oracle) if p in oracle.screen]
    pred = model.predict(scale.transform(np.array([[oracle.screen[p]] for p in candidates])))
    return [candidates[i] for i in np.argsort(-pred, kind="stable")[:confirms]]


def point_metrics(problem, oracle, target, fold_info, step, spent, policy):
    model = problem.model(oracle)
    test = problem.test_rows
    pred = np.asarray(model.predict(problem.x[test]), float)
    med = problem.median(oracle)
    position = problem.pos[test]
    y = problem.y[test]
    d = pd.DataFrame({"position": position, "error": np.abs(y-pred),
                      "median_error": np.abs(y-med)})
    metric = {**fold_info, "target": target, "policy": policy, "step": step,
              "cost": spent, "vti_positions": len(oracle.vti),
              "baseline_only_positions": len(set(oracle.screen)-set(oracle.vti)),
              "position_mae": float(d.groupby("position").error.mean().mean()),
              "median_mae": float(d.groupby("position").median_error.mean().mean()),
              "high40_discovered": int(sum(np.mean(z[1]) >= 40 for z in oracle.vti.values()))}
    rows = [{**fold_info, "target": target, "policy": policy, "step": step,
             "cost": spent, "variant_key": problem.keys[i], "position": int(problem.pos[i]),
             "observed": float(problem.y[i]), "pred": float(v), "median_pred": med}
            for i, v in zip(test, pred)]
    return metric, rows


def simulate(d, x, folds, esm, args):
    policies = ["random", "diversity", "model_high", "uncertainty", "baseline_screen"]
    if esm is not None:
        policies.append("esmrank_diversity")
    if args.batch * args.baseline_cost % args.vti_cost:
        raise ValueError("Choose costs/batch so screening round cost matches direct round exactly")
    confirms = args.batch - args.batch * args.baseline_cost // args.vti_cost
    if confirms < 1:
        raise ValueError("Baseline screens too expensive relative to VTI")
    print(f"Policies: {policies}")
    print(f"Hypothetical costs: baseline={args.baseline_cost}, VTI={args.vti_cost}; "
          f"direct: {args.batch} VTI/round; screened: {args.batch} baseline + {confirms} VTI/round")
    metrics, heldout_rows = [], []
    fold_list = folds[:1] if args.quick else folds
    for fi, (rep, fold, trkeys, tekeys) in enumerate(fold_list, 1):
        problem = Problem(d, x, trkeys, tekeys, args.target, esm)
        if len(problem.train_pos) < args.seed_positions + args.steps * args.batch:
            raise ValueError("Insufficient train positions for simulation design")
        for draw in range(1 if args.quick else args.draws):
            rng_seed = np.random.default_rng(32000 + int(rep)*100 + int(fold)*10 + draw)
            seed_positions = list(map(int, rng_seed.choice(problem.train_pos,
                                      size=args.seed_positions, replace=False)))
            for policy in policies:
                rng = np.random.default_rng(38000 + int(rep)*1000 + int(fold)*100 + draw*10)
                oracle = problem.new_oracle()
                for p in seed_positions:
                    oracle.buy_vti(p)
                spent = args.seed_positions * args.vti_cost
                fold_info = {"repeat": int(rep), "fold": int(fold), "draw": draw}
                for step in range(args.steps+1):
                    met, preds = point_metrics(problem, oracle, args.target, fold_info,
                                               step, spent, policy)
                    metrics.append(met)
                    heldout_rows.extend(preds)
                    if step == args.steps:
                        break
                    if policy == "baseline_screen":
                        chosen = choose_screen(problem, oracle, args.batch, confirms)
                        spent += args.batch * args.baseline_cost
                    else:
                        chosen = choose_direct(policy, problem, oracle, rng, args.batch)
                    for p in chosen:
                        oracle.buy_vti(p)
                    spent += len(chosen)*args.vti_cost
                    if spent != (args.seed_positions + (step+1)*args.batch)*args.vti_cost:
                        raise AssertionError("Cost budget mismatch across policies")
        print(f"Completed outer {fi}/{len(fold_list)}: repeat={rep} fold={fold}; "
              f"available TRAIN positions={len(problem.train_pos)}", flush=True)
    return pd.DataFrame(metrics), pd.DataFrame(heldout_rows)


def report(metrics, oof, args):
    tag = "quick" if args.quick else "full"
    OUT.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUT/f"phase32a_{args.target}_{tag}_metrics.csv", index=False)
    oof.to_csv(OUT/f"phase32a_{args.target}_{tag}_predictions.csv", index=False)
    if args.quick:
        print("PASS one-fold smoke test ONLY; not a performance conclusion")
        return
    # The same heldout protein position appears in each repeat/draw. Treat it as
    # ONE independent unit for paired comparisons, not multiple observations.
    curves = metrics.groupby(["target", "policy", "cost", "step"], as_index=False).agg(
        mean_position_mae=("position_mae", "mean"),
        mean_median_mae=("median_mae", "mean"),
        high40_found=("high40_discovered", "mean"),
        mean_vti_positions=("vti_positions", "mean"),
        mean_baseline_only=("baseline_only_positions", "mean"))
    curves.to_csv(OUT/f"phase32a_{args.target}_curves.csv", index=False)
    print("\nCOST-AWARE LEARNING AND DISCOVERY CURVES")
    print(curves.round(3).to_string(index=False))

    final = oof[oof.step.eq(args.steps)].copy()
    if final.empty:
        raise ValueError("Missing final prediction checkpoint")
    final["ae"] = np.abs(final.observed-final.pred)
    pos_error = final.groupby(["policy", "position"], as_index=False).ae.mean()
    pvt = pos_error.pivot(index="position",columns="policy",values="ae")
    reference = pvt["random"]
    if reference.size != 168 or pvt.isna().any().any():
        raise AssertionError("Expected exactly 168 jointly observed held-out protein positions")
    rng = np.random.default_rng(32033)
    rows=[]
    for name in pvt.columns:
        if name == "random": continue
        gain=(reference-pvt[name]).to_numpy()
        boot=rng.choice(gain, size=(3000,len(gain)),replace=True).mean(axis=1)
        rows.append({"policy":name,"gain_vs_random_pp":float(gain.mean()),
                     "ci95_low":float(np.quantile(boot,.025)),
                     "ci95_high":float(np.quantile(boot,.975)),
                     "n_independent_positions":len(gain)})
    results=pd.DataFrame(rows).sort_values("gain_vs_random_pp",ascending=False)
    results.to_csv(OUT/f"phase32a_{args.target}_paired_bootstrap.csv",index=False)
    print("\nFINAL-BUDGET PAIRED POSITION BOOTSTRAP vs RANDOM (positive = improvement)")
    print(results.round(3).to_string(index=False))
    print("\nPASS: retrospective VTI TRAIN-only simulation, no protected VTI VAL/TEST outcomes")
    print("NOTE: costs are hypothetical; baseline screening availability is counterfactual")


def synthetic():
    from sklearn.model_selection import GroupKFold
    rng=np.random.default_rng(32)
    positions=np.repeat(np.arange(1,111),2)
    n=len(positions)
    x=rng.normal(size=(n,80))
    x[:,68:76] = np.nan
    y=30+8*x[:,0]-5*x[:,2]+rng.normal(0,12,n)
    baseline=20+5*x[:,1]+rng.normal(0,3,n)
    d=pd.DataFrame({"variant_key":[f"v{i}" for i in range(n)],
                    "protein_position":positions,"vti_rescue_pp":y,
                    "corrector_rescue_pp":y*0.6,"baseline_fsk_pct_wt":baseline})
    folds=[]
    for fi,(a,b) in enumerate(GroupKFold(4).split(x,groups=positions),1):
        folds.append((1,fi,set(d.variant_key.iloc[a]),set(d.variant_key.iloc[b])))
    args=argparse.Namespace(target="total",quick=True,draws=1,seed_positions=20,
                            steps=1,batch=10,baseline_cost=1,vti_cost=5)
    metrics,oof=simulate(d,x,folds,rng.normal(size=n),args)
    assert len(metrics)==6*2 and (metrics.cost.isin([100,150])).all()
    assert metrics.groupby("step").cost.nunique().eq(1).all()
    assert (metrics.groupby("step").policy.nunique()==6).all()
    assert len(oof)>0
    print("PASS synthetic 32A: masked-label oracle; budget parity; two-stage screens; six policies")


def main():
    a=argparse.ArgumentParser()
    a.add_argument("mode", choices=["audit", "simulate", "self-test"])
    a.add_argument("--target", choices=TARGETS, default="total")
    a.add_argument("--quick", action="store_true")
    a.add_argument("--draws", type=int, default=1)
    a.add_argument("--seed-positions", type=int, default=20)
    a.add_argument("--steps", type=int, default=4)
    a.add_argument("--batch", type=int, default=10)
    a.add_argument("--baseline-cost", type=int, default=1)
    a.add_argument("--vti-cost", type=int, default=5)
    args=a.parse_args()
    if args.mode=="self-test":
        synthetic();return
    if args.draws<1 or args.draws>5 or args.steps<1 or args.steps>8 or args.batch<1:
        a.error("draws must be 1..5; steps 1..8; batch >=1")
    if args.baseline_cost<0 or args.vti_cost<1:
        a.error("Costs must be nonnegative with positive VTI cost")
    d,x,folds,esm,reference=get_real()
    print("PHASE 32A | ADAPTIVE COST-AWARE ASSAY SELECTION | TRAIN ONLY")
    print(f"Dataset: {len(d)} VTI TRAIN variants, {d.protein_position.nunique()} positions; "
          f"{len(folds)} frozen folds, source={reference}")
    print(f"ESMRank genotype-only ranking available: {esm is not None}")
    print("Unit of experimental selection: protein position (all its observed variants together)")
    print("Assay costs are hypothetical; not validated real prices")
    if args.mode=="audit":
        for rep,fold,tr,te in folds:
            print(f" repeat={rep} fold={fold} train_positions={len(set(d[d.variant_key.isin(tr)].protein_position))} "
                  f"held_positions={len(set(d[d.variant_key.isin(te)].protein_position))}")
        print("PASS metadata and simulation contract; no assay acquisitions performed")
        return
    metrics,oof=simulate(d,x,folds,esm,args)
    report(metrics,oof,args)

if __name__=="__main__":
    main()
