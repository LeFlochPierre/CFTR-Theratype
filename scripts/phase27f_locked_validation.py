#!/usr/bin/env python3
"""Phase 27F: predeclared secondary validation of VTI models.

Usage (run from repository root):
  python scripts/phase27f_locked_validation.py --self-test
  python scripts/phase27f_locked_validation.py predict
  python scripts/phase27f_locked_validation.py score --bootstraps 5000

PREDICT does not read VTI VAL outcomes. SCORE only reads VTI VAL outcomes,
never VTI TEST. Freeze predictions + source-code SHA256 before SCORE.
Existing VTI VAL was used earlier (26N): NOT untouched independent validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "results/biology/phase27f"
PRED = OUT / "phase27f_locked_predictions.csv"
MANIFEST = OUT / "phase27f_prediction_manifest.json"
SPLIT = ROOT / "data/processed/bihler_2026_transfer_splits.parquet"
VTI = ROOT / "data/processed/bihler_2026_vti.parquet"
EXPERT_ALPHA = 30.0
SOFT_GLOBAL_WEIGHT = 0.50
N_TRAIN, N_VAL = 210, 52
SEQ_N, STRUCT_N = 68, 12

# Frozen a priori: no tuning from Phase 27F validation outcomes.
CHOSEN = {
    "corrector_global": "phase26h fixed-spec XGB, 80 features + 1 constant",
    "acute": "phase26h fixed-spec XGB, same features, trained separately",
    "low_gate": "phase27d1 chemistry + XGB, corrector <10 pp",
    "high_gate": "phase27d3b sequence pocket + 3D heavy atom + logistic, corrector >40 pp",
    "expert": "phase27e1 seq68 StandardScaler+Ridge alpha=30, 3 response groups",
    "coupling": "phase27e2 heuristic binary coupling",
    "moe": "0.5 global + 0.5 soft weighted experts",
    "decision": "NO parameter, feature or threshold selection on validation labels",
}
CODE_DEPENDENCIES = [
    "scripts/phase27f_locked_validation.py",
    "scripts/phase26h_auxiliary_corrector.py",
    "scripts/phase27b1_extreme_classifiers.py",
    "scripts/phase27d1_biology_ablation.py",
    "scripts/phase27d3b_structure_ablation.py",
    "scripts/phase27e1_mixture_of_experts.py",
    "scripts/phase27e2_biological_gate_swap.py",
    "src/cftr_theratype/model_zoo.py",
    "src/cftr_theratype/benchmark.py",
    "src/cftr_theratype/structure_benchmark.py",
]


def digest(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def positions_mae(actual, pred, positions):
    t = pd.DataFrame({"position": positions, "loss": np.abs(np.asarray(actual, float) - np.asarray(pred, float))})
    return float(t.groupby("position").loss.mean().mean())


def response_classes(corrector):
    a = np.asarray(corrector, float)
    return np.where(a < 10, 0, np.where(a > 40, 2, 1)).astype(int)


def code_hashes():
    return {rel: digest(ROOT / rel) for rel in CODE_DEPENDENCIES if (ROOT / rel).is_file()}


def load_validation_metadata_only():
    """No labels; metadata ONLY from frozen partition file."""
    columns = ["variant_key", "protein_position", "transfer_split", "study_cohort", "variant", "variant_type", "hgvs_protein"]
    raw = pd.read_parquet(SPLIT, columns=columns)
    if raw.variant_key.isna().any() or raw.variant_key.duplicated().any():
        raise ValueError("Frozen metadata has ambiguous variant identities")
    val = raw.loc[raw.transfer_split.eq("val")].copy()
    if len(val) != N_VAL or val.protein_position.nunique() != 42:
        raise ValueError(f"Expected VTI VAL 52 variants / 42 positions; found {len(val)} / {val.protein_position.nunique()}")
    return val.sort_values("variant_key").reset_index(drop=True)


def make_validation_features(train, x_train80, validation):
    """Replicate model_zoo.prepare feature logic; enforce exact TRAIN parity."""
    from cftr_theratype.benchmark import make_features
    from cftr_theratype.reference import bihler_2024_assay_sequence
    from cftr_theratype.splits import parse_substitution
    from cftr_theratype import structure_benchmark as base

    seq = bihler_2024_assay_sequence()
    val = validation.copy()
    parsed = [parse_substitution(row, seq) for _, row in val.iterrows()]
    for p, position, key in zip(parsed, val.protein_position, val.variant_key):
        if len(p) < 4 or p[-1] != "valid" or int(p[0]) != int(position):
            raise ValueError(f"Validation amino-acid mapping failed for {key}: {p}")
    val["reference_aa"] = [str(p[1]) for p in parsed]
    val["alternate_aa"] = [str(p[2]) for p in parsed]
    val_codes = [f"{r}{int(p)}{a}" for r, p, a in zip(val.reference_aa, val.protein_position, val.alternate_aa)]
    if not all(re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]\d+[ACDEFGHIKLMNPQRSTVWY]", code) for code in val_codes):
        raise ValueError("Non-missense validation variant or unknown amino acid")

    tr_seq, train_names = make_features(train, seq)
    val_seq, val_names = make_features(val, seq)
    tr_seq, val_seq = np.asarray(tr_seq, float), np.asarray(val_seq, float)
    if tr_seq.shape != (N_TRAIN, SEQ_N) or val_seq.shape != (N_VAL, SEQ_N):
        raise ValueError(f"Unexpected sequence dimensions: train {tr_seq.shape}, val {val_seq.shape}")
    if list(train_names) != list(val_names):
        raise ValueError("Validation sequence feature names/order differ from TRAIN")
    if not np.allclose(tr_seq, x_train80[:, :SEQ_N], equal_nan=True, atol=1e-10, rtol=1e-10):
        raise ValueError("Feature parity failure: recreated TRAIN sequence differs from h.load_data")

    table = base.read_feature_table(
        ROOT / "results/structure/phase15b_position_features.csv",
        ROOT / "results/structure/phase15b_feature_summary.json", ROOT
    )
    tr_struct = np.asarray(base.structural_matrix(train.protein_position.to_numpy(int), table), float)
    val_struct = np.asarray(base.structural_matrix(val.protein_position.to_numpy(int), table), float)
    tr80 = np.column_stack([tr_seq, tr_struct])
    val80 = np.column_stack([val_seq, val_struct])
    if tr80.shape != (N_TRAIN, 80) or val80.shape != (N_VAL, 80):
        raise ValueError(f"Unexpected structural dimensions {tr80.shape} {val80.shape}")
    if not np.allclose(tr80, x_train80, equal_nan=True, atol=1e-10, rtol=1e-10):
        diff = np.nanmax(np.abs(tr80-x_train80))
        raise ValueError(f"Feature parity failure: TRAIN 80-D differs: max diff={diff}")
    if not np.isfinite(val80[:, :SEQ_N]).all() or np.isinf(val80).any():
        raise ValueError("Invalid VAL features (NaN allowed only in structural block)")
    print(f"PASS model_zoo feature parity: train80 {tr80.shape} identical, val80 {val80.shape}")
    print(f"Validation structural NaNs: {np.isnan(val80[:, SEQ_N:]).sum()}")
    return val, val_codes, val80


def frozen_cv_parity(v, x_train, yc, ya, folds):
    """Check refitting strategy against the original saved Phase26H2 OOF.

    This reads TRAIN-only OOF predictions. No validation/test labels.
    """
    import phase26h_auxiliary_corrector as h
    path = ROOT / "results/biology/phase26h2/phase26h2_oof.csv"
    if not path.is_file():
        print("WARN: Phase26H2 OOF not found; cannot verify acute-model parity")
        return {"corrector_max_abs": None, "acute_max_abs": None}
    ref = pd.read_csv(path, dtype={"variant_key": str})
    needed = {"repeat", "fold", "variant_key", "vti_fixed", "vti_fixed_total"}
    if not needed.issubset(ref) or len(ref) != 2*N_TRAIN:
        raise ValueError("Historical TRAIN OOF reference malformed")
    keys = v.variant_key.astype(str).to_numpy()
    positions = v.protein_position.to_numpy(int)
    x = np.column_stack([x_train, np.ones(N_TRAIN)])
    corrector_err, acute_err = [], []
    for repeat, fold, train_keys, test_keys in folds:
        it = np.flatnonzero(np.isin(keys, list(train_keys)))
        iv = np.flatnonzero(np.isin(keys, list(test_keys)))
        if len(it)+len(iv) != N_TRAIN or set(positions[it]) & set(positions[iv]):
            raise ValueError("Historical CV fold position leakage")
        sub = ref.loc[(ref.repeat==repeat) & (ref.fold==fold)].set_index("variant_key")
        if set(sub.index) != set(keys[iv]):
            raise ValueError("Historical OOF fold membership mismatch")
        sub = sub.loc[keys[iv]]
        mc, ma = h.previous.model(h.SPEC), h.previous.model(h.SPEC)
        mc.fit(x[it], yc[it]); ma.fit(x[it], ya[it])
        corrector_err.append(float(np.max(np.abs(mc.predict(x[iv])-sub.vti_fixed.to_numpy(float)))))
        acute_ref = (sub.vti_fixed_total-sub.vti_fixed).to_numpy(float)
        acute_err.append(float(np.max(np.abs(ma.predict(x[iv])-acute_ref))))
    result = {"corrector_max_abs": float(max(corrector_err)),
              "acute_max_abs": float(max(acute_err))}
    print("TRAIN OOF refit parity: corrector max |diff| =",result["corrector_max_abs"],
          "; acute max |diff| =",result["acute_max_abs"])
    if result["corrector_max_abs"] > 1e-5:
        raise ValueError("Corrector refit fails original TRAIN OOF parity; do not proceed")
    if result["acute_max_abs"] > 1e-5:
        print("WARN: acute refit differs from Phase26H2 reference; this validation uses a newly fit acute model")
    return result


def refit_global_and_acute(x_train, x_val, yc, ya):
    import phase26h_auxiliary_corrector as h
    a_train = np.column_stack([x_train, np.ones(len(x_train))])
    a_val = np.column_stack([x_val, np.ones(len(x_val))])
    corr = h.previous.model(h.SPEC)
    corr.fit(a_train, yc)
    acute = h.previous.model(h.SPEC)
    acute.fit(a_train, ya)
    pc, pa = np.asarray(corr.predict(a_val), float), np.asarray(acute.predict(a_val), float)
    if pc.shape != (N_VAL,) or pa.shape != (N_VAL,) or not np.isfinite(pc).all() or not np.isfinite(pa).all():
        raise ValueError("Invalid global/acute predictions")
    return pc, pa


def fit_gates(train, val, codes_train, codes_val, x_train, x_val, yc):
    import phase27b1_extreme_classifiers as b
    import phase27d1_biology_ablation as d1
    import phase27d3b_structure_ablation as d3
    d3.verify_provenance()
    geom = d3.load_geometry()
    a_train = d1.annotation_features(train.protein_position.to_numpy(int), codes_train)
    a_val = d1.annotation_features(val.protein_position.to_numpy(int), codes_val)
    p_tr = a_train["pocket_proxy"].to_numpy(float)
    p_va = a_val["pocket_proxy"].to_numpy(float)
    c_tr = a_train["chemistry"].to_numpy(float)
    c_va = a_val["chemistry"].to_numpy(float)
    f_tr, _ = d3.feature_sets(geom, train.protein_position.to_numpy(int), p_tr, c_tr)
    f_va, _ = d3.feature_sets(geom, val.protein_position.to_numpy(int), p_va, c_va)

    # Exact feature configurations selected before validation was scored.
    lo_train = np.column_stack([x_train, c_tr])
    lo_val = np.column_stack([x_val, c_va])
    hi_train = np.column_stack([x_train[:, :SEQ_N], f_tr["seq_plus_3d"]])
    hi_val = np.column_stack([x_val[:, :SEQ_N], f_va["seq_plus_3d"]])
    if not np.isfinite(hi_train).all() or not np.isfinite(hi_val).all():
        raise ValueError("Nonfinite high-gate feature vector")
    low = b.make_model("xgb_seq80", (yc < 10).astype(int), 27001)
    low.fit(lo_train, (yc < 10).astype(int))
    high = b.make_model("logistic_seq68", (yc > 40).astype(int), 27101)
    high.fit(hi_train, (yc > 40).astype(int))
    pl = np.asarray(low.predict_proba(lo_val)[:, 1], float)
    ph = np.asarray(high.predict_proba(hi_val)[:, 1], float)
    if (not np.isfinite(pl).all() or not np.isfinite(ph).all()
            or (pl < 0).any() or (ph < 0).any() or (pl > 1).any() or (ph > 1).any()):
        raise ValueError("Invalid predicted low/high scores")
    return pl, ph


def fit_experts(x_train, x_val, yc):
    # Use the Phase 27E-1 spec with all original 210 VTI TRAIN records.
    expert = np.zeros((len(x_val), 3), float)
    classes = response_classes(yc)
    counts = np.bincount(classes, minlength=3)
    for c in range(3):
        ix = np.flatnonzero(classes == c)
        if len(ix) < 10:
            raise ValueError("Too few training variants in an expert")
        model = make_pipeline(StandardScaler(), Ridge(alpha=EXPERT_ALPHA))
        model.fit(x_train[ix, :SEQ_N], yc[ix])
        expert[:, c] = model.predict(x_val[:, :SEQ_N])
    expert[:, 0] = np.minimum(expert[:, 0], 10.0)
    expert[:, 1] = np.clip(expert[:, 1], 10.0, 40.0)
    expert[:, 2] = np.maximum(expert[:, 2], 40.0)
    if not np.isfinite(expert).all():
        raise ValueError("Invalid expert predictions")
    print("Regime-expert training counts:", counts.tolist())
    return expert


def run_predict():
    if PRED.exists() or MANIFEST.exists():
        raise FileExistsError("LOCKED prediction or manifest already exists. Do not regenerate after examining validation results.")
    import phase26h_auxiliary_corrector as h
    import phase27d1_biology_ablation as d1
    import phase27e2_biological_gate_swap as e2

    print("=" * 104)
    print("PHASE 27F-1 | PREDICTION LOCK | VTI VAL OUTCOMES NOT OPENED")
    print("=" * 104)
    v, x80, _, _, folds = h.load_data()
    x80 = np.asarray(x80, float)
    if x80.shape != (N_TRAIN, 80) or len(v) != N_TRAIN or v.variant_key.nunique() != N_TRAIN:
        raise ValueError("Unexpected VTI TRAIN cohort")
    pos = v.protein_position.to_numpy(int)
    if len(np.unique(pos)) != 168 or len(folds) != 10:
        raise ValueError("Unexpected VTI TRAIN positions/folds")
    yc = v.corrector_rescue_pp.to_numpy(float)
    ya = v.vti_rescue_pp.to_numpy(float) - yc
    if not np.isfinite(yc).all() or not np.isfinite(ya).all():
        raise ValueError("Bad TRAIN outcomes")
    val_meta = load_validation_metadata_only()
    if set(pos) & set(val_meta.protein_position):
        raise ValueError("Protein-position overlap between TRAIN and VAL")
    v_val, val_codes, x_val = make_validation_features(v, x80, val_meta)
    train_codes = d1.extract_codes(v, pos)
    parity = frozen_cv_parity(v, x80, yc, ya, folds)
    pc, pa = refit_global_and_acute(x80, x_val, yc, ya)
    low, high = fit_gates(v, v_val, train_codes, val_codes, x80, x_val, yc)
    gate = e2.couple_binary(low, high)
    expert = fit_experts(x80, x_val, yc)
    moe_soft = (gate * expert).sum(axis=1)
    moe_soft50 = SOFT_GLOBAL_WEIGHT * pc + (1-SOFT_GLOBAL_WEIGHT)*moe_soft
    out = pd.DataFrame({
        "variant_key": v_val.variant_key.astype(str),
        "protein_position": v_val.protein_position.astype(int),
        "hgvs_protein": v_val.hgvs_protein.astype(str),
        "score_low": low, "score_high": high,
        "gate_low": gate[:,0], "gate_middle": gate[:,1], "gate_high": gate[:,2],
        "expert_low": expert[:,0], "expert_middle": expert[:,1], "expert_high": expert[:,2],
        "pred_corrector_global": pc, "pred_acute": pa,
        "pred_corrector_bio_soft": moe_soft,
        "pred_corrector_bio_soft50": moe_soft50,
        "pred_total_global": pc+pa,
        "pred_total_bio_soft": moe_soft+pa,
        "pred_total_bio_soft50": moe_soft50+pa,
    })
    if len(out) != N_VAL or out.variant_key.duplicated().any():
        raise ValueError("Duplicate/incomplete validation predictions")
    if not np.isfinite(out.select_dtypes(include=[np.number]).to_numpy()).all():
        raise ValueError("Nonfinite predictions")
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(PRED, index=False)
    manifest = {
        "phase": "27F_locked_secondary_validation",
        "cohort": "VTI VAL previously used in Phase26N; NOT independent untouched validation",
        "n_train": N_TRAIN, "n_val": N_VAL,
        "val_positions": int(v_val.protein_position.nunique()),
        "validation_outcomes_read": False,
        "heldout_test_outcomes_read": False,
        "pred_sha256": digest(PRED),
        "validation_id_digest": hashlib.sha256("\n".join(v_val.variant_key).encode()).hexdigest(),
        "source_code_sha256": code_hashes(),
        "training_oof_refit_parity": parity,
        "config": CHOSEN,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print("PASS: predictions locked BEFORE VAL labels were accessed")
    print("VAL identity:", len(out), "variants /", out.protein_position.nunique(), "positions")
    print("Prediction hash SHA256:", manifest["pred_sha256"])
    print("Output:", PRED)
    print("Next command: python scripts/phase27f_locked_validation.py score --bootstraps 5000")


def position_bootstrap_ap_gain(data, ycol, candidate, reference, nboot, seed):
    dd = data[["protein_position", ycol, candidate, reference]].copy()
    y = dd[ycol].to_numpy(int)
    a = dd[candidate].to_numpy(float)
    b = dd[reference].to_numpy(float)
    observed = float(average_precision_score(y,a)-average_precision_score(y,b))
    groups = [np.flatnonzero(dd.protein_position.to_numpy(int)==p) for p in sorted(dd.protein_position.unique())]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(nboot):
        ii = np.concatenate([groups[j] for j in rng.integers(0, len(groups), size=len(groups))])
        if len(np.unique(y[ii])) != 2:
            continue
        draws.append(float(average_precision_score(y[ii],a[ii])-average_precision_score(y[ii],b[ii])))
    if len(draws) < nboot * .8:
        return observed, np.nan, np.nan, len(draws)
    return observed, float(np.quantile(draws,.025)), float(np.quantile(draws,.975)), len(draws)


def position_bootstrap_mae_gain(data, label, candidate, reference, nboot, seed):
    df = data[["protein_position", label, candidate, reference]].copy()
    gains = (df.assign(gain=np.abs(df[reference]-df[label])-np.abs(df[candidate]-df[label]))
               .groupby("protein_position").gain.mean().to_numpy(float))
    rng = np.random.default_rng(seed)
    boot = gains[rng.integers(len(gains),size=(nboot,len(gains)))].mean(axis=1)
    return float(gains.mean()), float(np.quantile(boot,.025)),float(np.quantile(boot,.975))


def run_score(nboot):
    if not PRED.is_file() or not MANIFEST.is_file():
        raise FileNotFoundError("Run `predict` to lock predictions before scoring validation labels")
    with MANIFEST.open() as f:
        meta = json.load(f)
    if digest(PRED) != meta.get("pred_sha256"):
        raise ValueError("Prediction hash changed after lock — refusing evaluation")
    if meta.get("config") != CHOSEN:
        raise ValueError("Model configuration changed since locking predictions")
    if code_hashes() != meta.get("source_code_sha256"):
        raise ValueError("Dependency code differs from prediction lock")
    d = pd.read_csv(PRED, dtype={"variant_key": str})
    metadata = load_validation_metadata_only()
    if (len(d) != N_VAL or d.variant_key.duplicated().any() or
        set(d.variant_key) != set(metadata.variant_key) or
        not d.set_index("variant_key").loc[metadata.variant_key].protein_position.eq(metadata.protein_position.to_numpy(int)).all()):
        raise ValueError("Prediction vs frozen VAL metadata mismatch")
    keys = metadata.variant_key.astype(str).tolist()
    # Only two permitted VAL outcome columns, filtered to EXACT validation identities.
    assay = pd.read_parquet(VTI, columns=["variant_key","corrector_rescue_pp","vti_rescue_pp"],
                            filters=[("variant_key","in", keys)])
    if assay.variant_key.duplicated().any() or set(assay.variant_key) != set(keys) or len(assay) != N_VAL:
        raise ValueError("VAL assay membership differs from frozen partition")
    d = d.merge(assay.rename(columns={"corrector_rescue_pp":"observed_corrector", "vti_rescue_pp":"observed_total"}),
                on="variant_key", validate="one_to_one")
    if not np.isfinite(d[["observed_corrector","observed_total"]].to_numpy(float)).all():
        raise ValueError("Nonfinite VAL labels")
    c = response_classes(d.observed_corrector)
    d["class"] = c
    d["y_low"] = (c == 0).astype(int)
    d["y_high"] = (c == 2).astype(int)
    d["baseline_rank_low"] = -d.pred_corrector_global
    d["baseline_rank_high"] = d.pred_corrector_global
    print("="*108)
    print("PHASE 27F-2 | LOCKED SECONDARY VTI VALIDATION — READ VAL LABELS FOR SCORING ONLY")
    print("="*108)
    print("NOT untouched: VTI VAL was previously examined during Phase 26N")
    print("VAL:",len(d),"variants /",d.protein_position.nunique(),"positions")
    print("Corrector regimes low/middle/high:",np.bincount(c,minlength=3).tolist())
    print("TEST labels unopened. Models not refit or retuned in this stage.")

    print("\nA. RESPONDER RANKING")
    rows=[]
    for name, target, score, base in [("low", "y_low", "score_low", "baseline_rank_low"),
                                      ("high", "y_high", "score_high", "baseline_rank_high")]:
        y=d[target].to_numpy(int)
        if len(np.unique(y)) != 2:
            print(f"WARNING: {name}: one class only; AP/AUROC undefined")
            continue
        k = max(1,int(np.ceil(.20*len(d))))
        for label, col in [("global_regression_rank", base), ("biological_specialist", score)]:
            s=d[col].to_numpy(float)
            hits=int(y[np.argsort(-s,kind="stable")[:k]].sum())
            rows.append({"regime":name,"model":label,"positives":int(y.sum()),"ap":float(average_precision_score(y,s)),
                        "auroc":float(roc_auc_score(y,s)),"top20_selected":k,"top20_hits":hits,
                        "top20_precision":hits/k,"top20_recall":hits/int(y.sum())})
    ranking=pd.DataFrame(rows)
    print(ranking.round(4).to_string(index=False) if len(ranking) else "No task had both classes")

    print("\nB. REGRESSION MAE, POSITION-WEIGHTED")
    metrics=[]
    for target,label in [("corrector","observed_corrector"),("total","observed_total")]:
        for method in ["global","bio_soft","bio_soft50"]:
            col=f"pred_{target}_{method}"
            err=d[col]-d[label]
            metrics.append({"target":target,"model":method,"n":len(d),"positions":d.protein_position.nunique(),
                            "position_mae_pp":positions_mae(d[label],d[col],d.protein_position),
                            "variant_mae_pp":float(np.abs(err).mean()),"signed_bias_pp":float(err.mean())})
    mm=pd.DataFrame(metrics)
    print(mm.round(4).to_string(index=False))

    print("\nC. CORRECTOR MAE BY TRUE REGIME (diagnostic only)")
    subgroup=[]
    for c_id,name in enumerate(["low<10","middle10-40","high>40"]):
        q=d.loc[d["class"]==c_id]
        if q.empty: continue
        for method in ["global","bio_soft50"]:
            col=f"pred_corrector_{method}"
            subgroup.append({"regime":name,"model":method,"n":len(q),"positions":q.protein_position.nunique(),
                             "position_mae_pp":positions_mae(q.observed_corrector,q[col],q.protein_position),
                             "bias_pp":float((q[col]-q.observed_corrector).mean())})
    sub=pd.DataFrame(subgroup)
    print(sub.round(4).to_string(index=False))

    print("\nD. PAIRED POSITION-CLUSTERED DIFFERENCES")
    results=[]
    for i,(name,y,score,baseline) in enumerate([("low","y_low","score_low","baseline_rank_low"),
                                                  ("high","y_high","score_high","baseline_rank_high")]):
        if d[y].nunique() == 2:
            gain,lo,hi,draws=position_bootstrap_ap_gain(d,y,score,baseline,nboot,27001+i)
            results.append({"target":name,"metric":"AP gain specialist vs regression", "gain":gain,"ci95_low":lo,"ci95_high":hi,"bootstrap_valid":draws})
    for i,target in enumerate(["corrector","total"]):
        gain,lo,hi=position_bootstrap_mae_gain(d,f"observed_{target}",f"pred_{target}_bio_soft50",f"pred_{target}_global",nboot,27111+i)
        results.append({"target":target,"metric":"MAE reduction soft50 vs global (pp)","gain":gain,"ci95_low":lo,"ci95_high":hi,"bootstrap_valid":nboot})
    comparisons=pd.DataFrame(results)
    print(comparisons.round(4).to_string(index=False))

    print("\nE. GATE CONFUSION (diagnostic only)")
    gate=np.argmax(d[["gate_low","gate_middle","gate_high"]].to_numpy(float),axis=1)
    print("Actual low/mid/high x predicted low/mid/high")
    print(confusion_matrix(c, gate, labels=[0,1,2]))
    OUT.mkdir(parents=True,exist_ok=True)
    ranking.to_csv(OUT/"phase27f_val_ranking.csv",index=False)
    mm.to_csv(OUT/"phase27f_val_mae.csv",index=False)
    sub.to_csv(OUT/"phase27f_val_regimes.csv",index=False)
    comparisons.to_csv(OUT/"phase27f_val_bootstrap.csv",index=False)
    d.to_csv(OUT/"phase27f_val_scored_predictions.csv",index=False)
    print("\nPASS: Phase 27F secondary validation complete; outputs:",OUT)
    print("WARNING: not independent validation because 26N used VAL; no follow-up model selection from these labels.")


def self_test():
    y=np.array([0.,10.,40.,41.]); assert response_classes(y).tolist()==[0,1,1,2]
    q=pd.DataFrame({"protein_position":[1,1,2,3],"y":[1.,2.,3.,4.],"a":[2.,3.,4.,5.],"b":[1.,2.,3.,4.],"label":[0,0,1,1],"s1":[.1,.2,.8,.7],"s2":[.2,.3,.7,.6]})
    assert np.isclose(positions_mae(q.y,q.a,q.protein_position),1.0)
    a,b,c=position_bootstrap_mae_gain(q,"y","b","a",100,27)
    assert np.isclose(a,1.0) and b>=0 and c>=0
    z=position_bootstrap_ap_gain(q,"label","s1","s2",100,28)
    assert np.isfinite(z[0])
    print("PASS synthetic: class boundaries, position MAE, paired bootstraps")


if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("operation", nargs="?", choices=["predict","score"])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--bootstraps", type=int, default=5000)
    args=ap.parse_args()
    if args.self_test: self_test()
    elif args.operation == "predict": run_predict()
    elif args.operation == "score": run_score(args.bootstraps)
    else: ap.error("Choose predict or score; --self-test available")
