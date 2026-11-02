"""Phase 16B: map ThermoMPNN saturation predictions to frozen VTI identities.

No response/outcome tables are opened. ThermoMPNN position is zero-based and
relative to the first residue numbered in the PDB, NOT canonical CFTR position.
Its PDB parser represents gaps as '-' and preserves the integer residue span.

Usage:
  python -m cftr_theratype.thermompnn_adapter inspect
  python -m cftr_theratype.thermompnn_adapter collect \
    --predictions /path/to/ThermoMPNN_inference_6O2P_cftr_canonical_numbering.csv \
    --checkpoint ~/software/ThermoMPNN/models/thermoMPNN_default.pt \
    --source ~/software/ThermoMPNN
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import pandas as pd
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

ROOT = Path(__file__).resolve().parents[2]
PDB_REL = Path("results/structure/phase16_foldx/6O2P_cftr_canonical_numbering.pdb")
CANDIDATES_REL = Path("results/structure/phase16_foldx/candidates_train_only.csv")
OUTPUT_REL = Path("results/structure/phase16_thermompnn")
REQUIRED_COLUMNS = {"position", "wildtype", "mutation", "ddG_pred"}
EXPECTED_STUDY_COUNT = 210


def sha256(path: Path) -> str:
    hashobj = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hashobj.update(chunk)
    return hashobj.hexdigest()


def pdb_reference(pdb: Path, chain_id: str = "A") -> tuple[int, dict[int, str]]:
    """Return minimum *observed* residue number and observed 1-letter sequence.

    ThermoMPNN's alt_parse_PDB uses integer residue numbers and includes gaps
    across the full min..max span. Its 0-index position maps to pdb_min + pos.
    """
    chain = PDBParser(QUIET=True).get_structure("cftr", str(pdb))[0][chain_id]
    positions = {}
    for res in chain:
        if res.id[0] != " " or "CA" not in res:
            continue
        number, insertion = int(res.id[1]), str(res.id[2]).strip()
        if insertion:
            raise ValueError(f"Insertion code unsupported at residue {number}{insertion}")
        aa = seq1(res.get_resname(), custom_map={"UNK": "X"})
        if len(aa) != 1 or aa not in "ACDEFGHIKLMNPQRSTVWY":
            raise ValueError(f"Unknown amino acid at PDB position {number}: {res.resname}")
        if number in positions:
            raise ValueError(f"Duplicate structure position {number}")
        positions[number] = aa
    if not positions:
        raise ValueError("Prepared PDB has no CA coordinates")
    if len(positions) < 1100 and pdb.name.startswith("6O2P_"):
        raise ValueError("6O2P structural coverage suspiciously low")
    return min(positions), positions


def load_candidates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"variant_key", "hgvs_protein", "protein_position", "reference_aa", "alternate_aa", "status"}
    if not required <= set(df.columns):
        raise ValueError(f"Candidate manifest missing columns: {required - set(df.columns)}")
    if len(df) != EXPECTED_STUDY_COUNT or df.variant_key.duplicated().any():
        raise ValueError("Expected exactly 210 unique frozen VTI training identities")
    if (df["status"] == "eligible").sum() == 0:
        raise ValueError("No eligible variants found")
    return df


def join_predictions(pdb: Path, candidates: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Fail closed if the score-to-residue mapping cannot be verified."""
    missing = REQUIRED_COLUMNS - set(predictions.columns)
    if missing:
        raise ValueError(f"ThermoMPNN CSV missing: {missing}")
    df = predictions.copy()
    df["position"] = pd.to_numeric(df["position"], errors="raise").astype("int64")
    df["ddG_pred"] = pd.to_numeric(df["ddG_pred"], errors="coerce")
    if (df.position < 0).any():
        raise ValueError("Unexpected negative ThermoMPNN zero-based positions")
    if df[["position", "wildtype", "mutation"]].duplicated().any():
        raise ValueError("Duplicated ThermoMPNN mutation predictions")
    start, observed = pdb_reference(pdb)
    df["protein_position"] = df.position + start
    # Validate all scored residues, not only those in the VTI manifest.
    unknown = df.loc[~df.protein_position.isin(observed)]
    if not unknown.empty:
        raise ValueError(f"Predictions include unobserved PDB positions, e.g. {unknown.iloc[0].protein_position}")
    mismatch = df["wildtype"].astype(str).str.upper().ne(df.protein_position.map(observed))
    if mismatch.any():
        row = df.loc[mismatch].iloc[0]
        raise ValueError(f"Unsafe ThermoMPNN wildtype/position mapping at {row.protein_position}: "
                         f"CSV {row.wildtype} vs PDB {observed[int(row.protein_position)]}")
    bad_mut = ~df["mutation"].astype(str).str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]")
    if bad_mut.any():
        raise ValueError("ThermoMPNN predictions contain an invalid mutant residue")
    subset = df[["protein_position", "wildtype", "mutation", "ddG_pred"]].rename(
        columns={"wildtype": "reference_aa", "mutation": "alternate_aa", "ddG_pred": "thermompnn_ddg_pred"}
    )
    subset["reference_aa"] = subset["reference_aa"].astype(str)
    subset["alternate_aa"] = subset["alternate_aa"].astype(str)
    manifest = candidates.copy()
    manifest["protein_position"] = pd.to_numeric(manifest.protein_position, errors="raise").astype("int64")
    # Restrict to *eligible* identities; excluded identities remain with explicit missing scores.
    matched = manifest.merge(subset, on=["protein_position", "reference_aa", "alternate_aa"],
                             how="left", validate="many_to_one", indicator="_match")
    ineligible = matched.status.ne("eligible")
    matched.loc[ineligible, "thermompnn_ddg_pred"] = float("nan")
    matched["thermompnn_status"] = "not_structurally_eligible"
    matched.loc[~ineligible, "thermompnn_status"] = "missing_prediction"
    valid = (~ineligible) & matched["thermompnn_ddg_pred"].map(math.isfinite)
    matched.loc[valid, "thermompnn_status"] = "predicted"
    # A model failure must not be silently accepted as an ordinary missing structural coordinate.
    missing_eligible = matched.loc[(~ineligible) & (~valid)]
    if not missing_eligible.empty:
        example = missing_eligible.iloc[0]
        raise ValueError(f"No finite ThermoMPNN prediction for eligible {example.hgvs_protein}")
    # This file must NEVER include functional outcomes or study cohort metadata.
    keep = ["variant_key", "hgvs_protein", "protein_position", "reference_aa", "alternate_aa",
            "status", "thermompnn_ddg_pred", "thermompnn_status"]
    return matched[keep].sort_values(["protein_position", "variant_key"]).reset_index(drop=True)


def inspect(root: Path) -> None:
    pdb = root / PDB_REL
    candidates = load_candidates(root / CANDIDATES_REL)
    start, observed = pdb_reference(pdb)
    eligible = candidates.loc[candidates.status.eq("eligible")]
    print("=" * 78)
    print("CFTR-THERATYPE | THERMOMPNN REFERENCE / ELIGIBILITY CHECK")
    print("=" * 78)
    print(f"Reference PDB: {pdb}")
    print(f"First observed PDB residue: {start}; observed CA positions: {len(observed)}")
    print(f"Frozen VTI training: {len(candidates)} identities; eligible: {len(eligible)}")
    print(f"Status distribution: {candidates.status.value_counts().to_dict()}")
    print("Prediction CSV position i -> PDB canonical residue i + first observed PDB residue")
    print("CAUTION: 6O2P V470 and engineered E1371Q differ from functional assay background")
    print("No drug-binding energy or assay response values are used")
    print("=" * 78)


def git_hash(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def collect(root: Path, prediction_path: Path, checkpoint: Path, source: Path) -> None:
    pdb = root / PDB_REL
    candidates_path = root / CANDIDATES_REL
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Pretrained model checkpoint missing: {checkpoint}")
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    out = join_predictions(pdb, load_candidates(candidates_path), pd.read_csv(prediction_path))
    outdir = root / OUTPUT_REL
    outdir.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": "phase16b_thermompnn_single_mutation_reference_6O2P_v1",
        "prediction_file_sha256": sha256(prediction_path),
        "pdb_sha256": sha256(pdb),
        "candidate_manifest_sha256": sha256(candidates_path),
        "thermompnn_pretrained_checkpoint_sha256": sha256(checkpoint),
        "thermompnn_source_git_commit": git_hash(source),
        "n_train_variants": len(out),
        "n_scored": int(out.thermompnn_status.eq("predicted").sum()),
        "n_missing_structure": int(out.thermompnn_status.eq("not_structurally_eligible").sum()),
        "n_outcome_rows_accessed": 0,
        "sign_note": "ThermoMPNN predicted ddG sign must be verified against upstream documentation; reference construct only",
        "construct_note": "6O2P V470 and engineered E1371Q; unlike M470 in assay",
        "description": "Structure-specific predictive stability feature, not binding affinity or modulator response",
    }
    csv = outdir / "vti_train_stability_features.csv"
    manifest = outdir / "thermompnn_provenance.json"
    out.to_csv(csv, index=False)
    manifest.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("="*78)
    print("CFTR-THERATYPE | THERMOMPNN MUTATION-SPECIFIC FEATURES (TRAIN ONLY)")
    print("="*78)
    for key,val in report.items():
        print(f"  {key}: {val}")
    print("  Pilot preview:")
    print(out.loc[out.thermompnn_status.eq("predicted"),
                  ["hgvs_protein", "thermompnn_ddg_pred"]].head(12).to_string(index=False))
    print(f"Saved: {csv}")
    print(f"Saved: {manifest}")
    print("No assay outcomes were read; validation/test labels remain sealed")
    print("="*78)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["inspect", "collect"])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    if args.action == "inspect":
        inspect(args.root)
    else:
        if not all((args.predictions, args.checkpoint, args.source)):
            parser.error("collect requires --predictions, --checkpoint and --source")
        collect(args.root, args.predictions.expanduser().resolve(),
                args.checkpoint.expanduser().resolve(), args.source.expanduser().resolve())


if __name__ == "__main__":
    main()
