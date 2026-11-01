from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")

from phase27b3_threeclass import evaluate

OUT = Path("results/biology/phase27b")

d = pd.read_csv(
    OUT / "phase27b1_repeated_oof.csv",
    dtype={"variant_key": str}
)

KEY = [
    "repeat", "fold",
    "variant_key", "protein_position"
]

lo = (
    d.loc[d.task.eq("corrector_low"),
          KEY + ["label", "xgb_seq80"]]
    .rename(columns={
        "label": "is_low",
        "xgb_seq80": "low_score"
    })
)

hi = (
    d.loc[d.task.eq("corrector_high"),
          KEY + ["label", "logistic_seq68"]]
    .rename(columns={
        "label": "is_high",
        "logistic_seq68": "high_score"
    })
)

m = lo.merge(hi, on=KEY, validate="one_to_one")

assert len(m) == 420
assert not m.duplicated(KEY).any()
assert not ((m.is_low == 1) & (m.is_high == 1)).any()

m["true_class"] = np.where(
    m.is_low == 1, 0,
    np.where(m.is_high == 1, 2, 1)
)

low = np.clip(m.low_score.to_numpy(float), 0, 1)
high = np.clip(m.high_score.to_numpy(float), 0, 1)

total = low + high
scale = np.maximum(1.0, total)

# Preserve each specialist score wherever their sum <= 1.
p = np.column_stack([
    low / scale,
    np.maximum(0, 1 - total) / scale,
    high / scale
])

assert np.isfinite(p).all()
assert np.allclose(p.sum(axis=1), 1)
assert (p >= 0).all()

m["p_low"] = p[:, 0]
m["p_middle"] = p[:, 1]
m["p_high"] = p[:, 2]

avg = m.groupby(
    ["variant_key", "protein_position", "true_class"],
    as_index=False
)[["p_low", "p_middle", "p_high"]].mean()

assert len(avg) == 210

old = pd.read_csv(
    OUT / "phase27b3_averaged_oof.csv",
    dtype={"variant_key": str}
)

print("=" * 92)
print("PHASE 27B-3B | SPECIALIST-PRESERVING THREE-CLASS COMBINATION")
print("=" * 92)

print("OOF records:", len(m))
print("Unique variants:", len(avg))
print(
    "Cases requiring probability projection:",
    int((total > 1).sum()), "/", len(total)
)

print("\nA. MODEL COMPARISON")

models = {
    "specialist_projection": avg
}

for name in [
    "coupled_binary",
    "multinomial_logistic",
    "multiclass_xgb"
]:
    models[name] = old.loc[
        old.model.eq(name)
    ].copy()

rows = []

for name, q in models.items():
    result = evaluate(q)

    rows.append({
        "model": name,
        "balanced_accuracy": result["balanced_accuracy"],
        "macro_f1": result["macro_f1"],
        "low_ap": result["low_ap"],
        "middle_ap": result["middle_ap"],
        "high_ap": result["high_ap"],
        "low_top20": result["low_top20_hits"],
        "high_top20": result["high_top20_hits"],
        "log_loss": result["log_loss"],
        "brier": result["multiclass_brier"]
    })

summary = pd.DataFrame(rows)

print(summary.round(4).to_string(index=False))

print("\nB. SPECIALIST COMBINATION CONFUSION MATRIX")

from sklearn.metrics import confusion_matrix

prediction = p.argmax(axis=1)

# Evaluate unique variants using averaged probabilities.
pa = avg[["p_low", "p_middle", "p_high"]].to_numpy()
ya = avg.true_class.to_numpy(int)

print(
    confusion_matrix(
        ya,
        pa.argmax(axis=1),
        labels=[0, 1, 2]
    )
)

print("\nC. STANDALONE CLASSIFIER RANKING")

from sklearn.metrics import average_precision_score

print(
    "Low XGB AP:",
    round(
        average_precision_score(
            ya == 0, avg.p_low
        ), 4
    )
)

print(
    "High logistic AP:",
    round(
        average_precision_score(
            ya == 2, avg.p_high
        ), 4
    )
)

avg.to_csv(
    OUT / "phase27b3b_specialist_oof.csv",
    index=False
)

summary.to_csv(
    OUT / "phase27b3b_comparison.csv",
    index=False
)

print("\nPASS: No fitting; TRAIN OOF predictions only.")
