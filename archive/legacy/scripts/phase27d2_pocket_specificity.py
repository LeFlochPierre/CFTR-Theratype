#!/usr/bin/env python3
"""Phase 27D-2: is pocket-proxy improvement more than broad CFTR location?

VTI TRAIN only, exact original 2x5 grouped folds. Uses D1 frozen OOF and
compares sequence-distance pocket anchors against region-matched random anchors.
Negative controls are exploratory, not formal null-hypothesis tests.
No protected validation/test outcomes are accessed.
"""
from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))
import phase27d1_biology_ablation as d1
import phase27b1_extreme_classifiers as b

OUT = ROOT / "results/biology/phase27d"
FROZEN = OUT / "phase27d1_repeated_oof.csv"
KEY = ["repeat", "fold", "variant_key", "protein_position"]


def pocket_features(positions, anchors):
    pp = np.asarray(positions, int)
    cols = []
    for name in ("typeI_corrector_seq", "VX445_corrector_seq"):
        sites = np.asarray(anchors[name], int)
        dd = np.min(np.abs(pp[:, None] - sites[None, :]), axis=1)
        cols.extend([np.exp(-dd / 30), (dd <= 25).astype(float), dd / 100])
    return np.column_stack(cols)


def positional_controls(positions):
    p = np.asarray(positions, float) / 1480.0
    return np.column_stack([
        p, p*p, p*p*p,
        np.sin(2*np.pi*p), np.cos(2*np.pi*p),
        np.sin(4*np.pi*p), np.cos(4*np.pi*p)
    ])


def draw_decoy_anchors(rng):
    # Mirror each published-contact set's allocation to broad CFTR regions.
    # These arbitrary anchors are NOT published corrector contact sites.
    lasso_a = rng.choice(np.arange(1, 81), size=2, replace=False).tolist()
    tmd1 = rng.choice(np.arange(81, 366), size=3, replace=False).tolist()
    lasso_b = rng.choice(np.arange(1, 81), size=2, replace=False).tolist()
    tmd2 = rng.choice(np.arange(859, 1156), size=2, replace=False).tolist()
    return {
        "typeI_corrector_seq": sorted(lasso_a + tmd1),
        "VX445_corrector_seq": sorted(lasso_b + tmd2)
    }


def evaluate_variant_table(q):
    if len(q) != 210 or q.variant_key.nunique() != 210:
        raise ValueError("Expected one OOF mean per training variant")
    y, s = q.label.to_numpy(int), q.score.to_numpy(float)
    if y.sum() != 29 or not np.isfinite(s).all():
        raise ValueError("Unexpected high-responder labels/scores")
    ordered = np.argsort(-s, kind="stable")
    return {
        "ap": float(average_precision_score(y, s)),
        "auroc": float(roc_auc_score(y, s)),
        "top42_hits": int(y[ordered[:42]].sum()),
    }


def self_test():
    pos = np.array([18, 74, 198, 1102, 1251], dtype=int)
    real = pocket_features(pos, d1.CONTACT_SETS)
    assert real.shape == (5, 6)
    assert real[0, 3] == 1  # VX445 contact 18
    assert real[2, 0] == 1  # type I contact 198
    p = positional_controls(pos)
    assert p.shape == (5, 7)
    assert np.isfinite(p).all()
    anchors = draw_decoy_anchors(np.random.default_rng(2702))
    assert len(anchors["typeI_corrector_seq"]) == 5
    assert len(anchors["VX445_corrector_seq"]) == 4
    mock = pd.DataFrame({
        "variant_key": [f"v{i}" for i in range(210)],
        "label": [1]*29 + [0]*181,
        "score": np.arange(210, 0, -1) / 210,
    })
    assert np.isclose(evaluate_variant_table(mock)["ap"], 1.0)
    print("PASS synthetic: anchors, position controls, score calculation")


def run(n_decoys, n_boot):
    import phase26h_auxiliary_corrector as h

    if not FROZEN.is_file():
        raise FileNotFoundError(f"Run phase27d1_biology_ablation.py first: {FROZEN}")
    v, X80, _, _, folds = h.load_data()
    X80 = np.asarray(X80, float)
    positions = v.protein_position.to_numpy(int)
    keys = v.variant_key.astype(str).to_numpy()
    y = (v.corrector_rescue_pp.to_numpy(float) > 40).astype(int)
    if X80.shape != (210, 80) or len(np.unique(positions)) != 168 or y.sum() != 29:
        raise ValueError("Changed VTI TRAIN data")
    if not np.isfinite(X80[:, :68]).all():
        raise ValueError("Nonfinite sequence features")

    frozen = pd.read_csv(FROZEN, dtype={"variant_key": str})
    frozen = frozen[frozen.task.eq("corrector_high")].copy()
    if len(frozen) != 5 * 420 or frozen.duplicated(KEY + ["config"]).any():
        raise ValueError("Incomplete Phase27D1 high responder OOF")

    domains = d1.annotation_features(positions, d1.extract_codes(v, positions))["domains"].to_numpy(float)
    true_pocket = pocket_features(positions, d1.CONTACT_SETS)
    pos = positional_controls(positions)
    rng = np.random.default_rng(2702026)
    controls = {
        "position_smooth": pos,
        "domains_plus_position": np.column_stack([domains, pos]),
        "domains_plus_real_pocket": np.column_stack([domains, true_pocket]),
    }
    anchors_for_record = []
    for j in range(n_decoys):
        anchors = draw_decoy_anchors(rng)
        name = f"matched_decoy_{j+1:03d}"
        controls[name] = pocket_features(positions, anchors)
        anchors_for_record.append({"config": name, "anchors": str(anchors)})

    print("="*100)
    print("PHASE 27D-2 | REAL CORRECTOR CONTACT PROXIES VS REGION-MATCHED DECOYS")
    print("="*100)
    print("TRAIN only: 210 variants/168 positions; 29 high responders; original 2x5 grouped folds")
    print("Fixed high-responder logistic architecture; sequence68 + candidate features")
    print("Random contact anchors matched by broad CFTR region; not 3D contacts")
    print("Number of decoy anchor sets:", n_decoys)
    print("Protected VTI validation/test outcomes unopened")

    rows = []
    for fold_id, (repeat, fold, tr_keys, te_keys) in enumerate(folds, 1):
        tr = np.flatnonzero(np.isin(keys, list(tr_keys)))
        te = np.flatnonzero(np.isin(keys, list(te_keys)))
        if len(tr)+len(te) != 210 or set(positions[tr]) & set(positions[te]):
            raise ValueError(f"Group leakage in fold {fold_id}")
        if y[tr].sum() == 0 or y[tr].sum() == len(tr):
            raise ValueError("One-class outer training set")
        for name, feat in controls.items():
            XX = np.column_stack([X80[:, :68], feat])
            mdl = b.make_model("logistic_seq68", y[tr], 27000 + 100 + fold_id)
            mdl.fit(XX[tr], y[tr])
            scores = mdl.predict_proba(XX[te])[:, 1]
            if not np.isfinite(scores).all():
                raise ValueError("Nonfinite OOF scores")
            rows.extend({
                "repeat": int(repeat), "fold": int(fold),
                "variant_key": keys[idx], "protein_position": int(positions[idx]),
                "label": int(y[idx]), "config": name, "score": float(scores[j])
            } for j, idx in enumerate(te))
        print(f"completed fold {fold_id}/{len(folds)} | {len(controls)} controls", flush=True)

    pred = pd.DataFrame(rows)
    if len(pred) != len(controls)*420 or pred.duplicated(KEY+["config"]).any():
        raise ValueError("Incomplete control OOF")
    combined = pd.concat([frozen, pred], ignore_index=True)
    if combined.duplicated(KEY + ["config"]).any():
        raise ValueError("Duplicate OOF entry after combining")
    if (combined.groupby("config").size() != 420).any():
        raise ValueError("Each model must have 420 predictions")
    avg = combined.groupby(["config", "variant_key", "protein_position"],
                           as_index=False).agg(label=("label", "first"), score=("score", "mean"))
    metrics = pd.DataFrame([
        {"config": name, **evaluate_variant_table(q)}
        for name,q in avg.groupby("config")
    ]).sort_values("ap", ascending=False)

    print("\nA. REFERENCE + LOCATION + TRUE POCKET CONTROLS")
    nondec = metrics.loc[~metrics.config.str.startswith("matched_decoy")]
    print(nondec.round(4).to_string(index=False))
    dec = metrics.loc[metrics.config.str.startswith("matched_decoy")]
    real_ap = float(metrics.set_index("config").loc["pocket_proxy", "ap"])
    print("\nB. MATCHED-DECOY ANCHOR DISTRIBUTION")
    print(f"Decoy AP median={dec.ap.median():.4f}, p05={dec.ap.quantile(.05):.4f}, "
          f"p95={dec.ap.quantile(.95):.4f}, max={dec.ap.max():.4f}")
    print(f"Real-pocket AP={real_ap:.4f}; decoy sets AP >= real = "
          f"{int((dec.ap >= real_ap).sum())}/{len(dec)}")
    print("NOTE: descriptive sensitivity analysis, NOT a confirmatory permutation p-value")

    # Position-cluster bootstrap paired to original baseline and the real pocket.
    indexed = avg.pivot(index=["variant_key", "protein_position", "label"],
                        columns="config", values="score").reset_index()
    if len(indexed) != 210 or indexed.isna().any().any():
        raise ValueError("Misaligned OOF predictions")
    names = ["domains", "pocket_proxy", "position_smooth",
             "domains_plus_position", "domains_plus_real_pocket"]
    gains = []
    for ix, name in enumerate(names):
        gains.append({"config": name,
                      **b.ap_gain_bootstrap(indexed, "baseline", name, n_boot, 27750+ix)})
    print("\nC. BOOTSTRAP GAIN VS ORIGINAL SPECIALIST")
    print(pd.DataFrame(gains).round(4).to_string(index=False))
    OUT.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT / "phase27d2_repeated_oof.csv", index=False)
    avg.to_csv(OUT / "phase27d2_averaged_oof.csv", index=False)
    metrics.to_csv(OUT / "phase27d2_metrics.csv", index=False)
    pd.DataFrame(anchors_for_record).to_csv(OUT / "phase27d2_decoy_anchors.csv", index=False)
    pd.DataFrame(gains).to_csv(OUT / "phase27d2_ap_gain_bootstrap.csv", index=False)
    print("\nPASS: Phase27D-2 TRAIN-only control audit completed")
    print("No model selected on protected validation/test data. Outputs:", OUT)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--decoys", type=int, default=30)
    p.add_argument("--bootstraps", type=int, default=2000)
    a = p.parse_args()
    if a.self_test:
        self_test()
    else:
        if a.decoys < 5 or a.bootstraps < 100:
            p.error("Minimum --decoys 5 and --bootstraps 100")
        run(a.decoys, a.bootstraps)
