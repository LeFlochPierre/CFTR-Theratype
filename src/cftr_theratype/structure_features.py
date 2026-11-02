"""Phase 15B: train/test-label-free reference CFTR structural descriptors.

    python -m cftr_theratype.structure_features

Requires the completed Phase 15A structural-coverage audit. Does NOT read
functional assay tables, OOF predictions, response labels, or model outputs.
Biopython is required, including the Shrake-Rupley solvent-accessibility tool.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Data.PDBData import residue_sasa_scales
from Bio.PDB import MMCIFParser, ShrakeRupley
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from cftr_theratype.structure_coverage import (
    ROOT, STRUCTURES, align_to_canonical, canonical_cftr_sequence,
    parse_coordinates, select_cftr_entity, sha256_file,
)

PHASE = "15B_structure_features"
N_SASA_POINTS = 100
CA_CONTACT_CUTOFF_A = 8.0
LONG_RANGE_SEQUENCE_SEPARATION = 100
PDB_IDS = tuple(STRUCTURES)
STRUCTURE_ONLY_COLS = (
    "reference_contacts_8A", "reference_long_range_contacts_8A",
    "reference_protein_sasa_A2", "reference_protein_relative_sasa",
    "6O2P_VX7_ca_distance_A", "8EIQ_VX7_ca_distance_A",
    "8EIQ_CV6_ca_distance_A", "8EIQ_WJX_ca_distance_A",
)


def validate_source_files(root: Path, report: dict) -> None:
    """Refuse mismatched inputs: the structural map must come from these very CIFs/splits."""
    recorded = report.get("file_sha256")
    if not isinstance(recorded, dict):
        raise ValueError("Phase 15A summary has no input-file SHA256 provenance")
    for rel in (
        "data/external/structures/6O2P.cif",
        "data/external/structures/8EIQ.cif",
        "data/processed/bihler_2024_missense_splits.parquet",
        "data/processed/bihler_2026_transfer_splits.parquet",
    ):
        path = root / rel
        if not path.is_file() or recorded.get(rel) != sha256_file(path):
            raise ValueError(f"Phase 15A provenance mismatch/missing: {rel}; rerun structural coverage")
    sources = report.get("pdb_sources", {})
    for pdb in PDB_IDS:
        src = sources.get(pdb, {})
        if src.get("file_sha256") != recorded[f"data/external/structures/{pdb}.cif"]:
            raise ValueError(f"Phase 15A {pdb} source hash inconsistency")
        if src.get("ligands_without_coordinates"):
            raise ValueError(f"Missing expected experimental ligand coordinates: {pdb}")


def relative_sasa_for_chain(path: Path, chain_label: str) -> dict[int, tuple[float, float]]:
    """Protein-chain-only Shrake-Rupley SASA keyed by deposited label_seq_id.

    auth_chains=False + auth_residues=False makes parser residue indices
    compatible with Phase 15A label_asym_id and label_seq_id mapping.
    The membrane and ligands are absent from the SASA calculation. Consequently
    the values are protein-only *reference* exposure descriptors, not physical
    membrane-accessibility measurements or mutant structural predictions.
    """
    parser = MMCIFParser(QUIET=True, auth_chains=False, auth_residues=False)
    structure = parser.get_structure(path.stem, str(path))
    chains = {chain.id: chain for chain in structure[0]}
    if chain_label not in chains:
        raise ValueError(f"Missing label chain {chain_label} in {path.name}: {list(chains)}")
    chain = chains[chain_label]
    sr = ShrakeRupley(n_points=N_SASA_POINTS)
    sr.compute(chain, level="R")
    reference = residue_sasa_scales["Sander"]
    exposure = {}
    for residue in chain:
        if residue.id[0] != " ":
            continue
        deposited_index = residue.id[1]
        sasa = float(getattr(residue, "sasa", np.nan))
        maximum = reference.get(residue.resname.upper())
        relative = sasa / maximum if maximum is not None and maximum > 0 else np.nan
        if np.isfinite(sasa) and sasa >= 0:
            exposure[deposited_index] = (sasa, relative)
    if len(exposure) < 1000:
        raise ValueError(f"Too few resolved protein residues with SASA in {path.name}: {len(exposure)}")
    return exposure


def long_range_contacts(ca_by_ref: dict[int, np.ndarray], cutoff: float = CA_CONTACT_CUTOFF_A,
                        min_sequence_gap: int = LONG_RANGE_SEQUENCE_SEPARATION) -> dict[int, int]:
    """Number of CA neighbours within 8 A but >100 AA apart in sequence."""
    if not ca_by_ref:
        return {}
    indices = np.array(sorted(ca_by_ref), dtype=int)
    xyz = np.stack([ca_by_ref[i] for i in indices])
    result = {}
    for i, position in enumerate(indices):
        squared = np.einsum("ij,ij->i", xyz - xyz[i], xyz - xyz[i])
        result[int(position)] = int(((squared < cutoff ** 2) &
                                     (np.abs(indices - position) > min_sequence_gap)).sum())
    return result


def structure_geometry(root: Path, pdb_id: str, info: dict, canonical: str) -> dict[int, dict]:
    """Map SASA and 3D long-range contacts onto canonical protein numbering."""
    path = root / f"data/external/structures/{pdb_id}.cif"
    cif = MMCIF2Dict(str(path))
    entity, deposited = select_cftr_entity(cif, canonical)
    mapping, mismatches, identity = align_to_canonical(canonical, deposited)
    if identity < .99:
        raise ValueError(f"Unexpectedly weak {pdb_id} sequence alignment")
    chain_label, ca, _ = parse_coordinates(cif, entity)
    if chain_label != info["cftr_chain_label_asym_id"]:
        raise ValueError(f"Mismatch in selected CFTR chain for {pdb_id}")
    exposure = relative_sasa_for_chain(path, chain_label)
    coords = {ref: ca[dep] for ref, dep in mapping.items() if dep in ca}
    long_range = long_range_contacts(coords)
    out = {}
    for reference_pos, deposited_pos in mapping.items():
        if deposited_pos not in ca or reference_pos in mismatches:
            continue  # Only identical, coordinate-resolved canonical reference residues
        if deposited_pos not in exposure:
            raise ValueError(f"Missing SASA for mapped C-alpha residue {pdb_id}:{reference_pos}")
        abs_sasa, relative = exposure[deposited_pos]
        out[reference_pos] = {
            f"{pdb_id}_protein_sasa_A2": abs_sasa,
            f"{pdb_id}_protein_relative_sasa": relative,
            f"{pdb_id}_long_range_contacts_8A": long_range[reference_pos],
        }
    return out


def build_features(mapping: pd.DataFrame, geometries: dict[str, dict[int, dict]]) -> pd.DataFrame:
    required = {"protein_position", "6O2P_status", "8EIQ_status",
                "6O2P_ca_contacts_8A", "8EIQ_ca_contacts_8A",
                "6O2P_VX7_ca_distance_A", "8EIQ_VX7_ca_distance_A",
                "8EIQ_CV6_ca_distance_A", "8EIQ_WJX_ca_distance_A"}
    if not required.issubset(mapping.columns):
        raise ValueError(f"Missing Phase 15A structural columns: {sorted(required-set(mapping.columns))}")
    if mapping.protein_position.isna().any() or mapping.protein_position.duplicated().any():
        raise ValueError("Missing/duplicated protein positions in Phase 15A map")
    if not mapping.protein_position.between(1, 1480).all():
        raise ValueError("Invalid protein position in Phase 15A map")
    out = mapping.copy()
    out["protein_position"] = out.protein_position.astype(int)
    for pdb in PDB_IDS:
        exact = out[f"{pdb}_status"].eq("resolved_exact")
        out[f"{pdb}_resolved_exact"] = exact.astype(int)
        # Important: NEVER impute structural values using coordinates at a
        # construct-mismatched or unresolved reference residue.
        for key in ("ca_contacts_8A", *STRUCTURES[pdb]):
            col = (f"{pdb}_{key}" if key == "ca_contacts_8A"
                   else f"{pdb}_{key}_ca_distance_A")
            out.loc[~exact, col] = np.nan
        g = pd.DataFrame.from_records(
            [{"protein_position": pos, **rec} for pos, rec in geometries[pdb].items()]
        )
        if g.empty or g.protein_position.duplicated().any():
            raise ValueError(f"Invalid {pdb} structure geometry map")
        if set(g.protein_position) != set(out.loc[exact, "protein_position"]):
            diff = set(g.protein_position) ^ set(out.loc[exact, "protein_position"])
            # geometry contains all residues of the full CFTR structure; subset below
            if not set(out.loc[exact, "protein_position"]).issubset(set(g.protein_position)):
                raise ValueError(f"Incomplete {pdb} exact-position geometry: {sorted(diff)[:8]}")
        g = g.loc[g.protein_position.isin(out.protein_position)]
        out = out.merge(g, on="protein_position", how="left", validate="one_to_one")
        for col in (f"{pdb}_protein_sasa_A2", f"{pdb}_protein_relative_sasa",
                    f"{pdb}_long_range_contacts_8A"):
            out.loc[~exact, col] = np.nan
            if out.loc[exact, col].isna().any():
                raise ValueError(f"Incomplete {pdb} {col} for exact resolved positions")
    out["has_exact_reference"] = (out["6O2P_resolved_exact"] |
                                   out["8EIQ_resolved_exact"]).astype(int)
    out["primary_reference"] = np.select(
        [out["6O2P_resolved_exact"].eq(1), out["8EIQ_resolved_exact"].eq(1)],
        ["6O2P", "8EIQ"], default="unresolved")
    out["reference_contacts_8A"] = out["6O2P_ca_contacts_8A"].combine_first(out["8EIQ_ca_contacts_8A"])
    out["reference_long_range_contacts_8A"] = out["6O2P_long_range_contacts_8A"].combine_first(out["8EIQ_long_range_contacts_8A"])
    out["reference_protein_sasa_A2"] = out["6O2P_protein_sasa_A2"].combine_first(out["8EIQ_protein_sasa_A2"])
    out["reference_protein_relative_sasa"] = out["6O2P_protein_relative_sasa"].combine_first(out["8EIQ_protein_relative_sasa"])
    out["assay_reference_M470_vs_PDB_V470"] = out.protein_position.eq(470).astype(int)
    # No response or train/test partition columns are read or stored here.
    present = [c for c in STRUCTURE_ONLY_COLS if c in out.columns]
    if len(present) != len(STRUCTURE_ONLY_COLS):
        raise ValueError("Missing prespecified structural feature names")
    if out.loc[out.has_exact_reference.eq(0), list(STRUCTURE_ONLY_COLS)].notna().any().any():
        raise ValueError("Unresolved coordinates received non-null structural descriptors")
    return out.sort_values("protein_position").reset_index(drop=True)


def run(root: Path = ROOT) -> None:
    root = Path(root)
    reportfile = root / "results/structure/phase15_coverage_summary.json"
    infile = root / "results/structure/phase15_coverage_positions.csv"
    if not reportfile.exists() or not infile.exists():
        raise FileNotFoundError("Phase 15A outputs missing. Run python -m cftr_theratype.structure_coverage first")
    report = json.loads(reportfile.read_text())
    if report.get("phase") != "15A_structure_coverage" or report.get("outcome_data_accessed") is not False:
        raise ValueError("Unexpected Phase 15A summary; refuse to continue")
    validate_source_files(root, report)
    mapping = pd.read_csv(infile)
    canonical, _ = canonical_cftr_sequence()
    print("=" * 88)
    print("CFTR-THERATYPE | PHASE 15B REFERENCE STRUCTURAL FEATURES (LABEL-FREE)")
    print("=" * 88)
    geometry = {}
    for pdb in PDB_IDS:
        print(f"[1] Calculating {pdb} protein-only SASA and long-range CA contacts...")
        geometry[pdb] = structure_geometry(root, pdb, report["pdb_sources"][pdb], canonical)
        print(f"    {len(geometry[pdb])} exact-residue structural environments")
    features = build_features(mapping, geometry)
    covered = features.has_exact_reference.astype(bool)
    print("\n[2] EXACT REFERENCE MAPPING + FEATURE COVERAGE")
    print(f"    Distinct dataset positions: {len(features)}")
    print(f"    Covered by 6O2P and/or 8EIQ: {covered.sum()} ({covered.mean():.1%})")
    print(f"    Not exactly covered: {(~covered).sum()} — preserved as missing")
    print(f"    Primary 6O2P: {(features.primary_reference == '6O2P').sum()}")
    print(f"    Fallback 8EIQ: {(features.primary_reference == '8EIQ').sum()}")
    print("\n[3] DESCRIPTIVE VALUES (RESOLVED POSITIONS ONLY)")
    for col in ("reference_contacts_8A", "reference_long_range_contacts_8A",
                "reference_protein_sasa_A2", "reference_protein_relative_sasa",
                "6O2P_VX7_ca_distance_A", "8EIQ_CV6_ca_distance_A",
                "8EIQ_WJX_ca_distance_A"):
        values = pd.to_numeric(features.loc[covered, col], errors="coerce").dropna()
        med = values.median() if len(values) else float('nan')
        print(f"    {col:<43} available={len(values):>4} median={med:8.2f}")
    outdir = root / "results/structure"
    output = outdir / "phase15b_position_features.csv"
    summary = outdir / "phase15b_feature_summary.json"
    features.to_csv(output, index=False)
    metadata = {
        "phase": PHASE,
        "derived_from": "phase15_coverage_positions.csv and two experimental reference structures",
        "phase15a_summary_sha256": sha256_file(reportfile),
        "phase15a_position_map_sha256": sha256_file(infile),
        "structures_sha256": {p: sha256_file(root / f"data/external/structures/{p}.cif") for p in PDB_IDS},
        "number_positions": len(features), "positions_exactly_covered": int(covered.sum()),
        "structural_predictor_columns": list(STRUCTURE_ONLY_COLS) + [
            "has_exact_reference", "6O2P_resolved_exact", "8EIQ_resolved_exact",
            "assay_reference_M470_vs_PDB_V470",
        ],
        "measurement_provenance": "Experimental reference geometry, NOT variant-specific measurements",
        "sasa": {"algorithm": "Biopython ShrakeRupley", "points": N_SASA_POINTS,
                 "probe_A": 1.4, "protein_only": True, "normalization": "Sander"},
        "contacts": {"CA_cutoff_A": CA_CONTACT_CUTOFF_A,
                     "long_range_min_sequence_separation": LONG_RANGE_SEQUENCE_SEPARATION},
        "outcomes_loaded": False, "model_trained": False, "test_outcomes_loaded": False,
        "limitations": [
            "Protein-only SASA ignores membrane and ligands; hydrophobic exposure cannot be equated to aqueous solvent exposure",
            "8EIQ is a F508del construct, and 6O2P/8EIQ are different conformational and ligand states",
            "No mutant structures or mutation-induced energetic differences are modeled",
            "6O2P VX7 and 8EIQ VX7/CV6/WJX are experimental ligands, not vanzacaftor binding measurements",
            "Reference positions share structural features among distinct substitutions at the same residue",
            "Position 470 differs from assay M470 background, and 1371 may be construct-engineered",
            "Retain unresolved-position rows in all future split-based prediction analyses",
        ],
    }
    summary.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print("\n[4] SAVED")
    print(f"    {output.relative_to(root)}")
    print(f"    {summary.relative_to(root)}")
    print("PASS: No functional response labels, model outcomes, or assay tables accessed")
    print("=" * 88)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root")
    args = parser.parse_args()
    run(args.root)


if __name__ == "__main__":
    main()
