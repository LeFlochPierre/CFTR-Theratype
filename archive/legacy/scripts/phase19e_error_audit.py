"""CFTR Theratype Phase 19E: frozen VTI TRAIN-only acute IVA OOF error audit.

Descriptive diagnostics only: do NOT reuse assay outcomes as deployment-time features.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.cwd()
DATA = ROOT / "data/processed"
RESULTS = ROOT / "results/boltz"
OUT = RESULTS / "phase19e"
OUT.mkdir(parents=True, exist_ok=True)

BEST = "sequence_structure__xgboost"
SEQ = "sequence__xgboost"
MED = "median_baseline"
ET = "sequence_structure__extra_trees"
MODELS = [BEST, SEQ, MED, ET]

meta = pd.read_parquet(DATA / "bihler_2026_transfer_splits.parquet",
                       columns=["variant_key", "protein_position", "hgvs_protein", "study_cohort", "transfer_split"])
assert meta.variant_key.is_unique
train = meta.loc[meta.transfer_split.eq("train"),
                 ["variant_key", "protein_position", "hgvs_protein", "study_cohort"]].copy()
assert len(train) == 210 and train.protein_position.nunique() == 168

assay = pd.read_csv(RESULTS / "phase19b/vti_train_acute_iva_targets.csv",
                    dtype={"variant_key": str})
assay_cols = ["variant_key", "acute_iva_vti_pp", "acute_iva_dmso_pp", "acute_iva_change_pp",
              "corrector_fsk_gain_pp", "vti_rescue_pp", "fsk_sem_vnz_tez", "fsk_iva_sem_vnz_tez"]
assert assay.variant_key.is_unique and set(train.variant_key) == set(assay.variant_key)
assert set(assay_cols).issubset(assay.columns)

avg = pd.read_csv(RESULTS / "phase19c/acute_iva_vti_pp_averaged.csv",
                  dtype={"variant_key": str})
assert not avg.duplicated(["model", "variant_key"]).any()
for model in MODELS:
    sub = avg.loc[avg.model.eq(model)]
    assert len(sub) == 210 and set(sub.variant_key) == set(train.variant_key)
    assert sub.n_repeats.eq(2).all()

# The only outcomes read here come from the 210 frozen training variants.
frame = train.merge(assay[assay_cols], on="variant_key", validate="one_to_one")
wide = avg.pivot(index="variant_key", columns="model", values="predicted_pp")
frame = frame.merge(wide[MODELS].reset_index(), on="variant_key", validate="one_to_one")
obs = avg.loc[avg.model.eq(BEST), ["variant_key", "observed_pp"]]
check = frame.merge(obs, on="variant_key", validate="one_to_one")
assert np.allclose(check.acute_iva_vti_pp.to_numpy(float), check.observed_pp.to_numpy(float), atol=1e-8, rtol=0)
frame = check.drop(columns="observed_pp")

frame["true_pp"] = frame.acute_iva_vti_pp
frame["pred_best_pp"] = frame[BEST]
frame["pred_seq_pp"] = frame[SEQ]
frame["pred_median_pp"] = frame[MED]
for name, col in [("best", BEST), ("seq", SEQ), ("median", MED), ("struct_et", ET)]:
    frame[f"ae_{name}"] = (frame.true_pp - frame[col]).abs()
    frame[f"signed_{name}"] = frame.true_pp - frame[col]  # + means underprediction
frame["gain_best_vs_seq_pp"] = frame.ae_seq - frame.ae_best
frame["gain_best_vs_median_pp"] = frame.ae_median - frame.ae_best
frame["acute_sem_independent_proxy_pp"] = np.hypot(frame.fsk_sem_vnz_tez,
                                                   frame.fsk_iva_sem_vnz_tez)

# Two predictions from independent training-fold repeats: diagnostic instability, not uncertainty calibration.
oof = pd.read_csv(RESULTS / "phase19c/acute_iva_vti_pp_oof.csv",
                  dtype={"variant_key": str})
sub = oof.loc[oof.model.eq(BEST)].copy()
assert len(sub) == 420 and not sub.duplicated(["variant_key", "repeat"]).any()
wide_repeats = sub.pivot(index="variant_key", columns="repeat", values="predicted_pp")
assert set(wide_repeats.columns) == {1, 2} and len(wide_repeats) == 210
wide_repeats.columns = ["repeat_1_pp", "repeat_2_pp"]
frame = frame.merge(wide_repeats.reset_index(), on="variant_key", validate="one_to_one")
frame["repeat_disagreement_pp"] = (frame.repeat_1_pp - frame.repeat_2_pp).abs()
frame["repeat1_ae_pp"] = (frame.true_pp - frame.repeat_1_pp).abs()
frame["repeat2_ae_pp"] = (frame.true_pp - frame.repeat_2_pp).abs()

frame["response_bin"] = pd.cut(
    frame.true_pp,
    bins=[-np.inf, 2, 5, 10, 20, 40, np.inf],
    labels=["<=2", "2-5", "5-10", "10-20", "20-40", ">40"],
    include_lowest=True,
)
frame["corrector_gain_quartile"] = pd.qcut(frame.corrector_fsk_gain_pp, q=4, duplicates="drop").astype(str)
frame["dmso_iva_quartile"] = pd.qcut(frame.acute_iva_dmso_pp, q=4, duplicates="drop").astype(str)
frame["study_cohort"] = frame.study_cohort.fillna("missing").astype(str)


def grouped(column):
    t = frame.groupby(column, observed=True, dropna=False).agg(
        n_variants=("variant_key", "size"),
        n_positions=("protein_position", "nunique"),
        mean_true_pp=("true_pp", "mean"),
        mae_best_pp=("ae_best", "mean"),
        mae_sequence_pp=("ae_seq", "mean"),
        mae_median_pp=("ae_median", "mean"),
        mean_signed_error_pp=("signed_best", "mean"),
        mean_repeat_disagreement_pp=("repeat_disagreement_pp", "mean"),
    ).reset_index()
    t["gain_vs_sequence_pp"] = t.mae_sequence_pp - t.mae_best_pp
    t["gain_vs_median_pp"] = t.mae_median_pp - t.mae_best_pp
    return t


def display(label, table, columns=None, n=None):
    print("\n" + "=" * 115)
    print(label)
    print("=" * 115)
    if n is not None:
        table = table.head(n)
    if columns is not None:
        table = table[columns]
    print(table.to_string(index=False, float_format=lambda a: f"{a:.2f}"))


print("=" * 115)
print("PHASE 19E | TRAIN-ONLY ACUTE IVA OOF ERROR AUDIT")
print("=" * 115)
print("210 variants / 168 positions | repeated 2 x 5 position-disjoint CV | outcomes restricted to VTI TRAIN")
print("Signed error = observed - predicted; positive implies UNDERprediction")
print("Subgroup MAEs are variant-level, descriptive only; full benchmark uses position-macro MAE.")

worst_cols = ["variant_key", "hgvs_protein", "protein_position", "study_cohort",
              "true_pp", "pred_best_pp", "signed_best", "ae_best", "ae_seq",
              "ae_median", "repeat_disagreement_pp"]
display("TOP 20 WORST INDIVIDUAL VARIANTS | STRUCTURE + XGBOOST",
        frame.sort_values("ae_best", ascending=False), worst_cols, n=20)

bypos = frame.groupby("protein_position").agg(
    n_variants=("variant_key", "size"),
    true_mean_pp=("true_pp", "mean"),
    mae_best_pp=("ae_best", "mean"),
    mae_seq_pp=("ae_seq", "mean"),
    mae_median_pp=("ae_median", "mean"),
    signed_best_pp=("signed_best", "mean"),
).reset_index()
bypos["gain_vs_seq_pp"] = bypos.mae_seq_pp - bypos.mae_best_pp
bypos["gain_vs_median_pp"] = bypos.mae_median_pp - bypos.mae_best_pp
display("15 WORST PROTEIN POSITIONS (POSITION-LEVEL ERROR)",
        bypos.sort_values("mae_best_pp", ascending=False), n=15)

response = grouped("response_bin")
cohort = grouped("study_cohort")
corrector = grouped("corrector_gain_quartile")
dmso = grouped("dmso_iva_quartile")
for name, t in [
    ("ERROR BY MEASURED ACUTE IVA RESPONSE", response),
    ("ERROR BY STUDY COHORT", cohort),
    ("ERROR BY CORRECTOR-ASSOCIATED FSK GAIN QUARTILE", corrector),
    ("ERROR BY DMSO ACUTE IVA QUARTILE", dmso),
]:
    display(name, t)

print("\n" + "=" * 115)
print("DIAGNOSTIC SIGNALS")
print("=" * 115)
for target in ["true_pp", "corrector_fsk_gain_pp", "acute_iva_dmso_pp",
               "acute_sem_independent_proxy_pp", "repeat_disagreement_pp"]:
    rho = frame[target].corr(frame.ae_best, method="spearman")
    rho_signed = frame[target].corr(frame.signed_best, method="spearman")
    print(f"  {target:<36s} rho(|error|)={rho:+.3f}  rho(observed-predicted)={rho_signed:+.3f}")

for threshold in [5, 10, 20]:
    n = int((frame.ae_best >= threshold).sum())
    both = int(((frame.repeat1_ae_pp >= threshold) & (frame.repeat2_ae_pp >= threshold)).sum())
    print(f"  Error >= {threshold:>2} pp: {n:>3}/210 averaged OOF; {both:>3}/210 in BOTH CV repeats")
print("  Mean observed-predicted bias:  ", round(frame.signed_best.mean(), 3), "pp")
print("  Median observed-predicted bias:", round(frame.signed_best.median(), 3), "pp")

frame["min_repeat_error_pp"] = frame[["repeat1_ae_pp", "repeat2_ae_pp"]].min(axis=1)
display("REPEATEDLY HARD CASES | LARGE ERRORS IN BOTH CV REPEATS",
        frame.sort_values("min_repeat_error_pp", ascending=False),
        ["variant_key", "hgvs_protein", "true_pp", "pred_best_pp",
         "repeat1_ae_pp", "repeat2_ae_pp", "min_repeat_error_pp"], 12)

display("WHERE STATIC STRUCTURE HELPS MOST VS SEQUENCE-ONLY XGBOOST",
        frame.sort_values("gain_best_vs_seq_pp", ascending=False),
        ["variant_key", "hgvs_protein", "true_pp", "ae_best", "ae_seq", "gain_best_vs_seq_pp"], 12)
display("WHERE STATIC STRUCTURE HURTS MOST VS SEQUENCE-ONLY XGBOOST",
        frame.sort_values("gain_best_vs_seq_pp"),
        ["variant_key", "hgvs_protein", "true_pp", "ae_best", "ae_seq", "gain_best_vs_seq_pp"], 12)

pilot = ["c.922T>C", "c.941G>T", "c.1054C>G", "c.2750A>G", "c.4333G>A", "c.1897C>A"]
display("SIX BOLTZ PILOT VARIANTS | CURRENT MODEL ERRORS",
        frame.loc[frame.variant_key.isin(pilot)].sort_values("protein_position"),
        ["variant_key", "hgvs_protein", "true_pp", "pred_best_pp", "ae_best", "ae_seq", "ae_median"])

frame.sort_values("ae_best", ascending=False).to_csv(OUT / "annotated_oof_all210.csv", index=False)
frame.sort_values("ae_best", ascending=False).head(20).to_csv(OUT / "worst20_variants.csv", index=False)
bypos.sort_values("mae_best_pp", ascending=False).to_csv(OUT / "position_errors.csv", index=False)
response.to_csv(OUT / "by_response_bin.csv", index=False)
cohort.to_csv(OUT / "by_cohort.csv", index=False)
corrector.to_csv(OUT / "by_corrector_quartile.csv", index=False)
dmso.to_csv(OUT / "by_dmso_iva_quartile.csv", index=False)

feature_file = ROOT / "results/structure/phase15b_position_features.csv"
if feature_file.exists():
    print("\nAVAILABLE STATIC STRUCTURE COLUMNS:")
    print("  ", ", ".join(pd.read_csv(feature_file, nrows=0).columns))

print("\nQC PASS: 210/210 VTI TRAIN identities; observed OOF targets verified; val/test outcomes not opened")
print("Saved diagnostics in:", OUT)
print("NOTE: Stratification by assay outcomes is for error diagnosis, NOT deployment-time input features.")
print("NOTE: SEM proxy assumes independent FSK/FSK+IVA errors; use descriptively, not as exact assay uncertainty.")
