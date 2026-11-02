#!/usr/bin/env python3
"""Phase 30B: Incremental assay value, TRAIN OOF only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path("results/biology/phase30a/phase30a_oof.csv")
OUT = Path("results/biology/phase30b")

OBS = [
    "observed_baseline",
    "observed_corrector",
    "observed_total_rescue",
    "observed_treated_absolute",
]

PAIRS = [
    ("baseline", "mutation_only", "train_median",
     "Genotype contribution to untreated function"),

    ("corrector", "mutation_plus_baseline", "baseline_only",
     "Mutation value conditional on baseline"),
    ("corrector", "mutation_plus_baseline", "mutation_only",
     "Baseline assay value"),
    ("corrector", "baseline_treated_anchor",
     "mutation_plus_baseline", "Anchor versus direct"),

    ("total_rescue", "mutation_plus_baseline", "baseline_only",
     "Mutation value conditional on baseline"),
    ("total_rescue", "mutation_plus_baseline", "mutation_only",
     "Baseline assay value"),
    ("total_rescue", "mutation_plus_baseline_iva",
     "mutation_plus_baseline", "IVA incremental value"),
    ("total_rescue", "mutation_plus_baseline_iva",
     "mutation_only", "Two assays versus mutation-only"),
    ("total_rescue", "baseline_treated_anchor",
     "mutation_plus_baseline", "Anchor versus direct"),

    ("treated_absolute", "mutation_plus_baseline", "baseline_only",
     "Mutation value conditional on baseline"),
    ("treated_absolute", "mutation_plus_baseline", "mutation_only",
     "Baseline assay value"),
    ("treated_absolute", "mutation_plus_baseline_iva",
     "mutation_plus_baseline", "IVA incremental value"),
    ("treated_absolute", "mutation_plus_baseline_iva",
     "mutation_only", "Two assays versus mutation-only"),
]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pcol(target, model):
    return f"pred_{target}__{model}"


def validate(o, strict=True):
    need = {
        "repeat", "fold", "variant_key",
        "protein_position", *OBS
    }

    for t, a, b, _ in PAIRS:
        need.update([pcol(t, a), pcol(t, b)])

    need.update([
        pcol("baseline", "mutation_only"),
        pcol("total_rescue", "baseline_treated_anchor"),
        pcol("treated_absolute", "mutation_plus_baseline"),
    ])

    absent = need - set(o.columns)
    if absent:
        raise ValueError(f"Missing fields: {sorted(absent)}")

    ids = ["repeat", "fold", "variant_key", "protein_position"]
    if o[ids].isna().any().any():
        raise ValueError("Missing identifiers")

    if o.duplicated(["repeat", "variant_key"]).any():
        raise ValueError("Duplicate OOF identifiers")

    if set(o.repeat.unique()) != {1, 2}:
        raise ValueError("Expected two repeats")

    byrep = {
        r: set(g.variant_key.astype(str))
        for r, g in o.groupby("repeat")
    }

    if byrep[1] != byrep[2]:
        raise ValueError("Different mutations across repeats")

    if (o.groupby("variant_key").protein_position.nunique() != 1).any():
        raise ValueError("Protein positions changed")

    if o.groupby(
        ["repeat", "protein_position"]
    ).fold.nunique().max() != 1:
        raise ValueError("Position split across folds")

    if strict:
        if (
            len(o) != 420
            or o.variant_key.nunique() != 210
            or o.protein_position.nunique() != 168
        ):
            raise ValueError("Frozen TRAIN structure changed")

    num = sorted(need - set(ids))
    if not np.isfinite(o[num].to_numpy(float)).all():
        raise ValueError("Nonfinite predictions or labels")

    for col in OBS:
        spread = o.groupby("variant_key")[col].agg(
            lambda x: float(x.max() - x.min())
        )
        if spread.max() > 1e-8:
            raise ValueError(f"Target changed across repeats: {col}")

    identity = (
        o.observed_treated_absolute
        - o.observed_baseline
        - o.observed_total_rescue
    )

    if abs(identity).max() > 1e-5:
        raise ValueError("Rescue arithmetic mismatch")

    anchor = (
        o[pcol("treated_absolute", "mutation_plus_baseline")]
        - o.observed_baseline
        - o[pcol("total_rescue", "baseline_treated_anchor")]
    )

    if abs(anchor).max() > 1e-5:
        raise ValueError("Anchored prediction mismatch")

    return {
        "rows": len(o),
        "variants": o.variant_key.nunique(),
        "positions": o.protein_position.nunique(),
    }


def position_mae(o, truth, pred):
    z = o[["repeat", "protein_position"]].copy()
    z["error"] = np.abs(
        o[truth].to_numpy(float)
        - o[pred].to_numpy(float)
    )

    return float(
        z.groupby(["repeat", "protein_position"])
        .error.mean()
        .groupby("protein_position")
        .mean()
        .mean()
    )


def positions_gain(o, target, candidate, reference, boot, seed):
    truth = o["observed_" + target].to_numpy(float)

    gain = (
        np.abs(truth - o[reference].to_numpy(float))
        - np.abs(truth - o[candidate].to_numpy(float))
    )

    z = o[["repeat", "protein_position"]].copy()
    z["gain"] = gain

    pos = (
        z.groupby(["repeat", "protein_position"])
        .gain.mean()
        .groupby("protein_position")
        .mean()
        .to_numpy(float)
    )

    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(pos), size=(boot, len(pos))
    )
    samples = pos[indices].mean(axis=1)

    return {
        "gain_pp": float(pos.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "n_positions": len(pos),
    }


def run(o, boot):
    qc = validate(o)

    print("=" * 112)
    print("PHASE 30B | INCREMENTAL ASSAY VALUE | TRAIN OOF ONLY")
    print("=" * 112)
    print(
        f"Rows={qc['rows']} | "
        f"Variants={qc['variants']} | "
        f"Positions={qc['positions']}"
    )
    print("No model refitting, validation/test labels or ETI data.")

    print("\nA. INCREMENTAL INFORMATION VALUE")
    results = []

    for i, (target, cand, ref, question) in enumerate(PAIRS):
        c = pcol(target, cand)
        r = pcol(target, ref)
        truth = "observed_" + target

        stat = positions_gain(
            o, target, c, r, boot, 3000 + i
        )

        results.append({
            "target": target,
            "comparison": f"{cand} vs {ref}",
            "question": question,
            "candidate_mae_pp": position_mae(o, truth, c),
            "reference_mae_pp": position_mae(o, truth, r),
            **stat,
        })

    rs = pd.DataFrame(results)

    print(
        rs[[
            "target", "comparison",
            "candidate_mae_pp", "reference_mae_pp",
            "gain_pp", "ci95_low", "ci95_high"
        ]].round(4).to_string(index=False)
    )

    d = o.copy()
    d["regime"] = np.select(
        [
            d.observed_corrector < 10,
            d.observed_corrector > 40,
        ],
        ["low", "high"],
        default="middle",
    )

    print("\nB. MAE BY TRUE CORRECTOR RESPONSE REGIME")

    reg = []

    configs = {
        "corrector": [
            "mutation_only",
            "baseline_only",
            "mutation_plus_baseline",
            "baseline_treated_anchor",
        ],
        "total_rescue": [
            "mutation_only",
            "mutation_plus_baseline",
            "mutation_plus_baseline_iva",
        ],
    }

    for regime, x in d.groupby("regime", sort=False):
        for target, models in configs.items():
            for model in models:
                reg.append({
                    "regime": regime,
                    "target": target,
                    "information": model,
                    "variants": x.variant_key.nunique(),
                    "positions": x.protein_position.nunique(),
                    "position_mae_pp": position_mae(
                        x, "observed_" + target,
                        pcol(target, model)
                    ),
                    "mean_prediction_pp": float(
                        x[pcol(target, model)].mean()
                    ),
                    "mean_truth_pp": float(
                        x["observed_" + target].mean()
                    ),
                })

    rg = pd.DataFrame(reg)
    print(rg.round(3).to_string(index=False))

    print("\nC. ERRORS BY BASELINE FUNCTION QUARTILE")

    uniq = d[d.repeat == 1].copy()

    _, edges = pd.qcut(
        uniq.observed_baseline,
        4,
        retbins=True,
        duplicates="drop",
    )

    if len(edges) != 5:
        raise ValueError("Cannot form four baseline quartiles")

    edges[0] = -np.inf
    edges[-1] = np.inf

    d["baseline_q"] = pd.cut(
        d.observed_baseline,
        bins=edges,
        labels=["Q1 low", "Q2", "Q3", "Q4 high"],
        include_lowest=True,
    )

    rows = []

    for quartile, x in d.groupby(
        "baseline_q", observed=True
    ):
        configs_q = {
            "corrector": [
                "mutation_only",
                "mutation_plus_baseline",
            ],
            "treated_absolute": [
                "mutation_only",
                "baseline_only",
                "mutation_plus_baseline",
                "mutation_plus_baseline_iva",
            ],
        }

        for target, models in configs_q.items():
            for model in models:
                rows.append({
                    "baseline_quartile": str(quartile),
                    "target": target,
                    "information": model,
                    "variants": x.variant_key.nunique(),
                    "positions": x.protein_position.nunique(),
                    "position_mae_pp": position_mae(
                        x, "observed_" + target,
                        pcol(target, model)
                    ),
                })

    qq = pd.DataFrame(rows)
    print(qq.round(3).to_string(index=False))

    print("\nD. PREDICTION DYNAMIC RANGE")

    assoc = []

    for name in [
        "baseline", "corrector",
        "total_rescue", "treated_absolute"
    ]:
        models = set()

        for target, candidate, reference, _ in PAIRS:
            if target == name:
                models.update([candidate, reference])

        for model in sorted(models):
            predictions = (
                d.groupby("variant_key")[pcol(name, model)]
                .mean()
            )
            truth = (
                d.groupby("variant_key")["observed_" + name]
                .first()
            )

            assoc.append({
                "target": name,
                "model": model,
                "pred_sd_pp": float(
                    predictions.std(ddof=0)
                ),
                "target_sd_pp": float(
                    truth.std(ddof=0)
                ),
                "spearman_pred_truth": float(
                    predictions.corr(truth, method="spearman")
                ),
            })

    ac = pd.DataFrame(assoc)
    print(ac.round(3).to_string(index=False))

    OUT.mkdir(parents=True, exist_ok=True)

    outputs = [
        ("incremental_gain", rs),
        ("by_regime", rg),
        ("by_baseline_quartile", qq),
        ("dynamic_range", ac),
    ]

    for filename, frame in outputs:
        frame.to_csv(
            OUT / f"phase30b_{filename}.csv",
            index=False,
        )

    provenance = {
        "source_oof": str(INPUT),
        "source_sha256": sha256(INPUT),
        "n_repeats": 2,
        "n_train_variants": qc["variants"],
        "n_positions": qc["positions"],
        "status": "Read-only exploratory TRAIN OOF",
        "warnings": [
            "Subtraction-derived target coupling",
            "Assay inputs required at deployment",
            "Bootstrap intervals not selection-adjusted",
        ],
    }

    (OUT / "phase30b_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )

    print("\nINTERPRETATION")
    print("Positive MAE gain means the candidate is better.")
    print("Models using measured assays are not mutation-only.")
    print("Baseline is mathematically coupled to rescue targets.")
    print("IVA-only assay is not an independent ETI assay.")
    print("All comparisons are exploratory TRAIN development.")

    print("\nPASS Phase 30B; saved:", OUT)


def self_test():
    rows = []

    for rep in [1, 2]:
        for i in range(12):
            baseline = 2 + i
            corrector = 4 + 3 * i
            total = corrector + 5
            treated = baseline + total

            row = {
                "repeat": rep,
                "fold": i // 3 + 1,
                "variant_key": f"key{i}",
                "protein_position": i,
                "observed_baseline": baseline,
                "observed_corrector": corrector,
                "observed_total_rescue": total,
                "observed_treated_absolute": treated,
            }

            for target, truth in [
                ("baseline", baseline),
                ("corrector", corrector),
                ("total_rescue", total),
                ("treated_absolute", treated),
            ]:
                for model in [
                    "train_median",
                    "mutation_only",
                    "baseline_only",
                    "mutation_plus_baseline",
                    "mutation_plus_baseline_iva",
                    "baseline_treated_anchor",
                ]:
                    row[pcol(target, model)] = (
                        truth + 2.0
                        if model == "mutation_only"
                        else truth + 1.0
                    )

            row[pcol(
                "total_rescue", "baseline_treated_anchor"
            )] = (
                row[pcol(
                    "treated_absolute",
                    "mutation_plus_baseline"
                )] - baseline
            )

            rows.append(row)

    o = pd.DataFrame(rows)
    qc = validate(o, strict=False)

    assert qc["positions"] == 12

    stat = positions_gain(
        o,
        "corrector",
        pcol("corrector", "mutation_plus_baseline"),
        pcol("corrector", "mutation_only"),
        400,
        0,
    )

    assert np.isclose(stat["gain_pp"], 1.0)

    bad = o.copy()
    bad.loc[0, "observed_treated_absolute"] += 1

    try:
        validate(bad, strict=False)
    except ValueError:
        pass
    else:
        raise AssertionError("Identity mismatch not rejected")

    print("PASS: synthetic Phase 30B tests")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--bootstraps", type=int, default=2000)

    args = parser.parse_args()

    if args.self_test:
        self_test()
    else:
        if args.bootstraps < 100:
            parser.error("Require >=100 bootstraps")

        self_test()

        if not INPUT.is_file():
            parser.error(
                f"Missing Phase 30A file: {INPUT}"
            )

        oof = pd.read_csv(
            INPUT,
            dtype={"variant_key": str},
        )

        run(oof, args.bootstraps)
