#!/usr/bin/env python3
"""Phase 26E: train-only, position-held-out TEZ-pocket structural-feature test.

Research question: does experimentally grounded proximity to the TEZ-bound CFTR
structure (PDB 7SV7) add information beyond existing mutation descriptors
for prediction of VNZ/TEZ corrector rescue and total VTI in-vitro rescue?

Requires Phase 26D and its completed phase26d_oof.csv.
Reuses the frozen OOF folds, outcomes and baseline predictions.
No protected VTI outcomes are read.

TEZ proximity is NOT a measured vanzacaftor binding descriptor.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


ROOT = Path.cwd()
OUT = ROOT / "results/biology/phase26e"
STRUCT = ROOT / "data/external/structures/7SV7.cif"
BASE_OOF = ROOT / "results/biology/phase26d/phase26d_oof.csv"
PDB_URL = "https://files.rcsb.org/download/7SV7.cif"


def pdb_features(path: Path, positions: np.ndarray, wt_aa: np.ndarray):
    """Extract TEZ proximity from PDB 7SV7 and audit residue numbering."""

    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    from Bio.SeqUtils import seq1

    cif = MMCIF2Dict(str(path))

    cols = [
        "group_PDB", "auth_asym_id", "auth_seq_id",
        "auth_comp_id", "label_atom_id", "label_comp_id",
        "type_symbol", "Cartn_x", "Cartn_y", "Cartn_z"
    ]

    a = {key: cif["_atom_site." + key] for key in cols}

    if len({len(v) for v in a.values()}) != 1:
        raise ValueError("Malformed atom-site table")

    residues = {}
    aa_by_pos = {}
    ligands = []

    for j in range(len(a["group_PDB"])):
        element = str(a["type_symbol"][j]).upper()

        if element in ("H", "D"):
            continue

        comp = a["label_comp_id"][j].upper()
        atom = a["label_atom_id"][j].upper()
        group = a["group_PDB"][j].upper()

        xyz = np.array(
            [float(a[k][j]) for k in ("Cartn_x", "Cartn_y", "Cartn_z")],
            dtype=float
        )

        if not np.isfinite(xyz).all():
            continue

        if group == "HETATM" and comp == "CV6":
            ligands.append(xyz)

        if group == "ATOM" and a["auth_asym_id"][j] == "A":
            try:
                pos = int(a["auth_seq_id"][j])
            except (ValueError, TypeError):
                continue

            residues.setdefault(pos, []).append((atom, xyz))

            if atom == "CA":
                aa_by_pos[pos] = seq1(
                    a["auth_comp_id"][j],
                    custom_map={"UNK": "X"}
                )

    if len(ligands) < 5 or len(residues) < 500:
        raise ValueError(
            f"Unexpected structure: ligand atoms={len(ligands)}, "
            f"CFTR residues={len(residues)}"
        )

    lig = np.array(ligands)

    valid_wt = [
        (int(p), str(w).strip().upper())
        for p, w in zip(positions, wt_aa)
        if int(p) in aa_by_pos
        and str(w).strip().upper() in "ACDEFGHIKLMNPQRSTVWY"
    ]

    n_match = sum(
        aa_by_pos[p].upper() == w
        for p, w in valid_wt
    )

    fraction = n_match / len(valid_wt) if valid_wt else 0

    print(
        f"PDB 7SV7 QC: ligand atoms={len(lig)}, "
        f"CFTR residues={len(residues)}, "
        f"AA matches={n_match}/{len(valid_wt)} "
        f"({fraction:.1%})"
    )

    if len(valid_wt) < 35 or fraction < 0.90:
        raise ValueError(
            "PDB residue numbering / wild-type identity validation failed. "
            "Do not use structural distances until mapping is corrected."
        )

    wrong_positions = {
        p for p, w in valid_wt
        if aa_by_pos[p].upper() != w
    }

    ca = {
        pos: next(
            (xyz for atom, xyz in atoms if atom == "CA"),
            None
        )
        for pos, atoms in residues.items()
    }

    ca = {p: xyz for p, xyz in ca.items() if xyz is not None}

    ca_pos = np.array(sorted(ca), dtype=int)
    ca_xyz = np.array([ca[p] for p in ca_pos])

    ligand_dist = {}

    for p, atoms in residues.items():
        xyz = np.array([x for _, x in atoms])

        ligand_dist[p] = float(
            np.linalg.norm(
                xyz[:, None, :] - lig[None, :, :],
                axis=2
            ).min()
        )

    pocket_positions = {
        p for p, dist in ligand_dist.items()
        if dist <= 8.0
    }

    pocket_ca = np.array([
        ca[p]
        for p in sorted(pocket_positions)
        if p in ca
    ])

    per_position = {}

    for p in np.unique(positions.astype(int)):
        if p in wrong_positions or p not in residues or p not in ca:
            per_position[p] = (np.nan, np.nan, np.nan)
            continue

        heavy_distance = ligand_dist[p]

        n_local = int(
            (
                np.linalg.norm(ca_xyz - ca[p], axis=1)
                <= 10.0
            ).sum()
        ) - 1

        pocket_distance = (
            float(
                np.linalg.norm(
                    pocket_ca - ca[p],
                    axis=1
                ).min()
            )
            if len(pocket_ca)
            else np.nan
        )

        per_position[p] = (
            heavy_distance,
            float(n_local),
            pocket_distance
        )

    feats = np.array([
        per_position[int(p)]
        for p in positions
    ], dtype=float)

    covered = int(np.isfinite(feats[:, 0]).sum())

    print(
        f"PDB feature coverage: {covered}/{len(positions)} variants"
    )

    print(
        "Features: TEZ atom distance, local CA packing, "
        "distance to TEZ-contacting residues"
    )

    if covered < max(50, int(0.3 * len(positions))):
        raise ValueError(
            "Too few aligned variants have coordinates."
        )

    return feats, per_position


def shuffle_at_position_level(pocket, positions, seed=2605):
    """Shuffle pocket descriptors between positions as a negative control."""

    rng = np.random.default_rng(seed)
    p = np.array(positions, dtype=int)
    unique = np.unique(p)

    valid = np.array([
        i for i, pos in enumerate(unique)
        if np.isfinite(
            pocket[np.flatnonzero(p == pos)[0], 0]
        )
    ], dtype=int)

    perm = valid.copy()
    rng.shuffle(perm)

    swapped = np.array(pocket, dtype=float, copy=True)

    for dst, src in zip(valid, perm):
        dst_ids = np.flatnonzero(p == unique[dst])
        src_id = np.flatnonzero(p == unique[src])[0]
        swapped[dst_ids] = pocket[src_id]

    assert np.array_equal(
        np.isnan(swapped),
        np.isnan(pocket)
    )

    return swapped


def fit_compare(d, X80, pocket, folds, baseline, quick=False):
    """Fit actual-pocket and shuffled-pocket models on frozen folds."""

    import phase26d_corrector_bottleneck as prev

    keys = d.variant_key.astype(str).to_numpy()
    pos = d.protein_position.to_numpy(int)

    y_corrector = d.corrector_rescue_pp.to_numpy(float)
    y_total = d.vti_rescue_pp.to_numpy(float)

    sham = shuffle_at_position_level(pocket, pos)

    mats = {
        "pocket_corrector": (
            np.column_stack((X80, pocket)),
            y_corrector
        ),
        "sham_corrector": (
            np.column_stack((X80, sham)),
            y_corrector
        ),
        "pocket_total": (
            np.column_stack((X80, pocket)),
            y_total
        ),
        "sham_total": (
            np.column_stack((X80, sham)),
            y_total
        ),
    }

    rows = []
    selections = []

    for z, (rep, fold, trkeys, tekeys) in enumerate(folds, 1):
        it = np.flatnonzero(np.isin(keys, list(trkeys)))
        iv = np.flatnonzero(np.isin(keys, list(tekeys)))

        assert set(pos[it]).isdisjoint(pos[iv])
        assert len(it) + len(iv) == len(d)

        fold_base = baseline[
            (baseline.repeat == rep)
            & (baseline.fold == fold)
        ]

        if set(fold_base.variant_key) != set(keys[iv]):
            raise ValueError(
                f"Phase26D baseline keys mismatch: "
                f"repeat={rep}, fold={fold}"
            )

        for name, (x, y) in mats.items():
            spec, inner = prev.select_spec(
                x[it], y[it], pos[it], quick=quick
            )

            mdl = prev.model(spec)
            mdl.fit(x[it], y[it])

            predictions = mdl.predict(x[iv])

            selections.append({
                "repeat": rep,
                "fold": fold,
                "model": name,
                "depth": spec[0],
                "lambda": spec[1],
                "n_trees": spec[2],
                "inner_mae": inner
            })

            for j, idx in enumerate(iv):
                rows.append({
                    "repeat": rep,
                    "fold": fold,
                    "variant_key": keys[idx],
                    "protein_position": int(pos[idx]),
                    "study_cohort": d.study_cohort.iloc[idx],
                    "observed_corrector": y_corrector[idx],
                    "observed_total": y_total[idx],
                    "model": name,
                    "prediction": float(predictions[j])
                })

        print(
            f"  completed {z}/{len(folds)}: "
            f"repeat={rep}, fold={fold}, "
            f"held-out positions={len(np.unique(pos[iv]))}",
            flush=True
        )

    wide = pd.DataFrame(rows).pivot(
        index=[
            "repeat", "fold", "variant_key",
            "protein_position", "study_cohort",
            "observed_corrector", "observed_total"
        ],
        columns="model",
        values="prediction"
    ).reset_index()

    wide.columns.name = None

    joined = baseline.merge(
        wide,
        on=[
            "repeat", "fold", "variant_key",
            "protein_position", "study_cohort"
        ],
        validate="one_to_one",
        suffixes=("_base", "")
    )

    if len(joined) != len(baseline):
        raise ValueError(
            "Missing Phase26D rows; refusing incomplete comparison"
        )

    for y in ("corrector", "total"):
        if not np.allclose(
            joined[f"observed_{y}_base"],
            joined[f"observed_{y}"]
        ):
            raise ValueError(
                f"Outcome mismatch relative to Phase26D: {y}"
            )

    joined["pocket_components_total"] = (
        joined.pocket_corrector
        + joined.pred_zero_acute_seq80
    )

    return joined, pd.DataFrame(selections)


def pos_mae(y, pred, group):
    t = pd.DataFrame({
        "position": group,
        "error": np.abs(np.asarray(y) - np.asarray(pred))
    })

    return float(t.groupby("position").error.mean().mean())


def evaluate(t, n_boot=5000):
    tasks = [
        ("Corrector", "pred_zero_corrector_seq80", "observed_corrector"),
        ("Corrector", "sham_corrector", "observed_corrector"),
        ("Corrector", "pocket_corrector", "observed_corrector"),
        ("Total", "pred_zero_total_seq80", "observed_total"),
        ("Total", "sham_total", "observed_total"),
        ("Total", "pocket_total", "observed_total"),
        ("Total", "pocket_components_total", "observed_total"),
    ]

    metrics = []

    for rep, df in t.groupby("repeat"):
        for task, model, target in tasks:
            y = df[target].to_numpy(float)
            pred = df[model].to_numpy(float)

            metrics.append({
                "repeat": rep,
                "target": task,
                "model": model,
                "position_mae_pp": pos_mae(
                    y, pred, df.protein_position
                ),
                "variant_mae_pp": float(
                    np.mean(np.abs(y - pred))
                ),
                "rmse_pp": float(
                    np.sqrt(np.mean((y - pred) ** 2))
                )
            })

    gains = []
    rng = np.random.default_rng(2605)

    comparisons = [
        (
            "pocket_corrector",
            "pred_zero_corrector_seq80",
            "observed_corrector"
        ),
        (
            "pocket_corrector",
            "sham_corrector",
            "observed_corrector"
        ),
        (
            "pocket_total",
            "pred_zero_total_seq80",
            "observed_total"
        ),
        (
            "pocket_total",
            "sham_total",
            "observed_total"
        ),
        (
            "pocket_components_total",
            "pred_zero_total_seq80",
            "observed_total"
        ),
    ]

    for cand, ref, target in comparisons:
        q = t.copy()

        q["gain"] = (
            np.abs(q[target] - q[ref])
            - np.abs(q[target] - q[cand])
        )

        z = (
            q.groupby(["repeat", "protein_position"])
            .gain.mean()
            .groupby("protein_position")
            .mean()
            .to_numpy()
        )

        sim = rng.choice(
            z,
            size=(n_boot, len(z)),
            replace=True
        ).mean(axis=1)

        gains.append({
            "candidate": cand,
            "reference": ref,
            "gain_mae_pp": float(z.mean()),
            "ci95_low": float(np.quantile(sim, 0.025)),
            "ci95_high": float(np.quantile(sim, 0.975)),
            "n_independent_positions": len(z)
        })

    return pd.DataFrame(metrics), pd.DataFrame(gains)


def self_test():
    import phase26d_corrector_bottleneck as prev

    rng = np.random.default_rng(2605)

    pos = np.repeat(np.arange(36), 2)
    n = len(pos)
    keys = np.array([f"syn{i}" for i in range(n)])

    X = rng.normal(size=(n, 80))
    X[0, 72] = np.nan

    p0 = rng.uniform(0, 30, size=36)

    pocket = np.column_stack([
        p0[pos],
        rng.uniform(2, 18, 36)[pos],
        rng.uniform(0, 18, 36)[pos]
    ])

    pocket[pos == 35] = np.nan

    d = pd.DataFrame({
        "variant_key": keys,
        "protein_position": pos,
        "study_cohort": ["syn"] * n
    })

    d["baseline_fsk_pct_wt"] = rng.uniform(0, 5, n)

    d["iva_treated_pct_wt"] = (
        d.baseline_fsk_pct_wt + rng.uniform(1, 8, n)
    )

    d["corrector_rescue_pp"] = (
        12
        + 0.1 * np.nan_to_num(pocket[:, 0])
        + 2 * np.nan_to_num(X[:, 0])
        + rng.normal(size=n)
    )

    d["vti_rescue_pp"] = (
        d.corrector_rescue_pp + 3 + rng.normal(size=n)
    )

    d["vti_acute_after_corrector_pp"] = (
        d.vti_rescue_pp - d.corrector_rescue_pp
    )

    folds = []

    for f, (it, iv) in enumerate(
        GroupKFold(3).split(X, groups=pos), 1
    ):
        folds.append(
            (1, f, set(keys[it]), set(keys[iv]))
        )

    oof, _ = prev.evaluate(
        d,
        {
            "sequence": X[:, :68],
            "sequence_structure": X
        },
        folds,
        quick=True
    )

    out, selection = fit_compare(
        d, X, pocket, folds, oof, quick=True
    )

    metrics, gains = evaluate(out, n_boot=100)

    assert len(out) == n
    assert len(selection) == len(folds) * 4
    assert np.isfinite(metrics.position_mae_pp).all()
    assert len(gains) == 5

    assert np.array_equal(
        np.isnan(shuffle_at_position_level(pocket, pos)),
        np.isnan(pocket)
    )

    print(
        "PASS: Phase26E synthetic; disjoint positions, "
        "baseline joins, missing coordinates, "
        "sham control, model fitting and bootstrap"
    )

    print(
        metrics.groupby(
            ["target", "model"], as_index=False
        ).position_mae_pp.mean().round(3).to_string(index=False)
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--structure", type=Path, default=STRUCT)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--bootstraps", type=int, default=5000)

    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    if not BASE_OOF.exists():
        raise FileNotFoundError(
            f"Need completed Phase26D OOF: {BASE_OOF}"
        )

    if not args.structure.exists():
        print(f"Downloading PDB 7SV7 to {args.structure} ...")

        args.structure.parent.mkdir(
            parents=True, exist_ok=True
        )

        try:
            urllib.request.urlretrieve(
                PDB_URL, str(args.structure)
            )
        except Exception as e:
            raise RuntimeError(
                f"Could not retrieve {PDB_URL}. "
                "Download 7SV7.cif from RCSB manually."
            ) from e

    import phase26d_corrector_bottleneck as prev

    d, feat, historical, folds, reference = prev.load_train()

    assert len(d) == 210
    assert d.protein_position.nunique() == 168

    print("=" * 98)
    print(
        "PHASE 26E | TEZACAFTOR-SITE STRUCTURAL FEATURES "
        "FOR VTI RESCUE | TRAIN ONLY"
    )
    print("=" * 98)

    print(
        "Hypothesis: TEZ-pocket proximity improves "
        "zero-shot VNZ/TEZ corrector prediction."
    )
    print(
        "7SV7 is TEZ-bound, NOT a structure of "
        "vanzacaftor binding."
    )
    print(
        "Same Phase26D frozen position-disjoint folds "
        "and baseline predictions."
    )
    print(
        "All 210 TRAIN variants; no VTI val/test "
        "outcomes accessed."
    )
    print(
        f"Historical reference: {reference} | "
        f"PDB: {args.structure}"
    )

    pocket, by_position = pdb_features(
        args.structure,
        d.protein_position.to_numpy(int),
        d.reference_aa.to_numpy(str)
    )

    oof = pd.read_csv(
        BASE_OOF,
        dtype={"variant_key": str}
    )

    X = np.asarray(
        feat["sequence_structure"],
        dtype=float
    ).copy()

    assert X.shape == (210, 80)

    X[np.isinf(X)] = np.nan

    results, sel = fit_compare(
        d, X, pocket, folds, oof,
        quick=args.quick
    )

    metrics, gains = evaluate(
        results, n_boot=args.bootstraps
    )

    feature_table = pd.DataFrame({
        "variant_key": d.variant_key,
        "protein_position": d.protein_position,
        "tez_min_atom_A": pocket[:, 0],
        "ca_neighbor_10A": pocket[:, 1],
        "tez_pocket_ca_distance_A": pocket[:, 2]
    })

    OUT.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        OUT / "phase26e_oof.csv", index=False
    )

    sel.to_csv(
        OUT / "phase26e_selections.csv", index=False
    )

    metrics.to_csv(
        OUT / "phase26e_metrics_by_repeat.csv", index=False
    )

    gains.to_csv(
        OUT / "phase26e_bootstrap.csv", index=False
    )

    feature_table.to_csv(
        OUT / "phase26e_drugsite_features.csv", index=False
    )

    print("\nA. MATCHED POSITION-LEVEL PREDICTION")

    summary = (
        metrics.groupby(
            ["target", "model"], as_index=False
        )[["position_mae_pp", "rmse_pp"]]
        .mean()
        .sort_values(["target", "position_mae_pp"])
    )

    print(
        summary.round(4).to_string(index=False)
    )

    print(
        "\nB. PAIRED POSITION BOOTSTRAP "
        "(positive = proposed model improved)"
    )

    print(
        gains.round(4).to_string(index=False)
    )

    print(
        "\nC. CORRECTOR AND TOTAL RESCUE "
        "HIGH-TAIL ERRORS (>40 pp)"
    )

    comparisons = [
        (
            "observed_corrector",
            "pred_zero_corrector_seq80",
            "pocket_corrector"
        ),
        (
            "observed_total",
            "pred_zero_total_seq80",
            "pocket_total"
        )
    ]

    for target, base, new in comparisons:
        s = results[results[target] > 40]

        print(
            f"{target}: "
            f"{len(s) // results.repeat.nunique()} "
            f"variants per repeat; "
            f"baseline MAE="
            f"{np.mean(np.abs(s[target] - s[base])):.3f}; "
            f"new MAE="
            f"{np.mean(np.abs(s[target] - s[new])):.3f}"
        )

    print("\nD. INNER MODEL CHOICES")

    print(
        sel.groupby(
            ["model", "depth", "lambda", "n_trees"]
        ).size().rename("outer_splits").to_string()
    )

    print(
        "\nPASS: Phase26E completed; outputs in",
        OUT
    )


if __name__ == "__main__":
    main()
