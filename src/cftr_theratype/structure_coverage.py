"""Phase 15: experimental CFTR structure coverage audit (identity metadata ONLY).

Usage: python -m cftr_theratype.structure_coverage --download

Reads split/variant-identity Parquet files, reference CFTR sequence, and RCSB
mmCIF structures. Never opens functional assay outcomes or held-out labels.
The coordinates are reference-construct environments, NOT mutant structures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner, substitution_matrices
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

ROOT = Path(__file__).resolve().parents[2]
STRUCTURES = {"6O2P": ["VX7"], "8EIQ": ["VX7", "CV6", "WJX"]}
PDB_URL = "https://files.rcsb.org/download/{pdb_id}.cif"
SOURCE_META = {
    "ETI": ("bihler_2024_missense_splits.parquet", "split", 586),
    "VTI": ("bihler_2026_transfer_splits.parquet", "transfer_split", 415),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_cftr_sequence() -> tuple[str, str]:
    """Derive canonical P13569 from the repo's experimentally grounded M470 reference."""
    from cftr_theratype.reference import bihler_2024_assay_sequence
    assay = str(bihler_2024_assay_sequence()).strip().upper()
    if len(assay) != 1480 or assay[469] != "M":
        raise ValueError("Unexpected M470 CFTR assay reference. Stop and verify sequence provenance.")
    canonical = assay[:469] + "V" + assay[470:]
    if canonical[507] != "F":
        raise ValueError("Unexpected residue at canonical F508. Check reference alignment.")
    return canonical, assay


def download_reference(path: Path, pdb_id: str) -> None:
    if path.exists():
        if path.stat().st_size < 100_000:
            raise ValueError(f"Existing structure is suspiciously small: {path}")
        print(f"  Existing {pdb_id}: {path} ({path.stat().st_size:,} bytes); preserving")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    url = PDB_URL.format(pdb_id=pdb_id)
    request = Request(url, headers={"User-Agent": "CFTR-Theratype-StructureAudit/1.0"})
    with urlopen(request, timeout=120) as response:
        data = response.read()
    if len(data) < 100_000 or b"_atom_site." not in data or b"_entity_poly." not in data:
        raise ValueError(f"Invalid or truncated mmCIF for {pdb_id} ({len(data)} bytes)")
    path.write_bytes(data)
    print(f"  Downloaded {pdb_id}: {len(data):,} bytes; sha256={sha256_file(path)}")


def _as_list(value):
    return value if isinstance(value, list) else [value]


def _column(cif: dict, key: str) -> list[str]:
    if key not in cif:
        raise ValueError(f"Missing mmCIF column: {key}")
    return _as_list(cif[key])


def select_cftr_entity(cif: dict, canonical: str) -> tuple[str, str]:
    """Find CFTR polymer by alignment-compatible long sequence, not chain 'A'."""
    ids = _column(cif, "_entity_poly.entity_id")
    seqs = _column(cif, "_entity_poly.pdbx_seq_one_letter_code_can")
    if len(ids) != len(seqs):
        raise ValueError("Malformed _entity_poly table")
    candidates = [(entity, re.sub(r"\s+", "", seq).upper())
                  for entity, seq in zip(ids, seqs)
                  if len(re.sub(r"\s+", "", seq)) >= 1000]
    if len(candidates) != 1:
        raise ValueError(f"Expected one full-length CFTR polymer, found {len(candidates)}")
    entity, seq = candidates[0]
    if len(seq) < 1300 or len(seq) > 1650:
        raise ValueError("Unexpected deposited CFTR sequence length")
    return entity, seq


def align_to_canonical(canonical: str, deposited: str, min_identity: float = 0.90):
    """Map 1-based UniProt positions to 1-based mmCIF label_seq_id indices."""
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    alignment = aligner.align(canonical, deposited)[0]
    mapping = {}
    mismatches = {}
    exact = 0
    for (ref_start, ref_end), (dep_start, dep_end) in zip(*alignment.aligned):
        if ref_end - ref_start != dep_end - dep_start:
            raise AssertionError("Unexpected nonaligned block lengths")
        for r, d in zip(range(ref_start, ref_end), range(dep_start, dep_end)):
            pos = r + 1
            mapping[pos] = d + 1
            if canonical[r] == deposited[d]:
                exact += 1
            else:
                mismatches[pos] = (canonical[r], deposited[d])
    identity = exact / max(1, len(mapping))
    if identity < min_identity:
        raise ValueError(f"Sequence alignment identity {identity:.3f} is too low; reject mapping")
    if len(mapping) < 1000:
        raise ValueError("Too few aligned CFTR residues; reject mapping")
    return mapping, mismatches, identity


def parse_coordinates(cif: dict, cftr_entity: str) -> tuple[str, dict[int, np.ndarray], dict[str, np.ndarray]]:
    """Use mmCIF label_seq_id for CFTR CA; retain heavy atoms for named drugs.

    Only first model and preferred altloc; never use author residue numbers.
    """
    fields = {key: _column(cif, "_atom_site." + key) for key in [
        "group_PDB", "label_entity_id", "label_asym_id", "label_seq_id",
        "label_atom_id", "label_comp_id", "type_symbol", "Cartn_x", "Cartn_y", "Cartn_z",
    ]}
    n = len(fields["group_PDB"])
    if any(len(v) != n for v in fields.values()):
        raise ValueError("Malformed atom-site table dimensions")
    modelnums = _as_list(cif.get("_atom_site.pdbx_PDB_model_num", ["1"] * n))
    alts = _as_list(cif.get("_atom_site.label_alt_id", ["."] * n))
    if len(modelnums) != n or len(alts) != n:
        raise ValueError("Malformed model/alternate-conformation fields")
    ca_by_chain: dict[str, dict[int, np.ndarray]] = {}
    ligand_coords: dict[str, list[np.ndarray]] = {}
    for i in range(n):
        if modelnums[i] != "1" or alts[i] not in (".", "?", "A", "1"):
            continue
        group = fields["group_PDB"][i]
        entity = fields["label_entity_id"][i]
        atom = fields["label_atom_id"][i]
        comp = fields["label_comp_id"][i]
        xyz = np.array([float(fields[k][i]) for k in ("Cartn_x", "Cartn_y", "Cartn_z")])
        if not np.isfinite(xyz).all():
            continue
        if (group == "ATOM" and entity == cftr_entity
                and atom == "CA" and fields["label_seq_id"][i] not in (".", "?")):
            chain = fields["label_asym_id"][i]
            pos = int(fields["label_seq_id"][i])
            ca_by_chain.setdefault(chain, {}).setdefault(pos, xyz)
        if group == "HETATM" and fields["type_symbol"][i].upper() not in ("H", "D"):
            ligand_coords.setdefault(comp, []).append(xyz)
    if not ca_by_chain:
        raise ValueError("No CFTR C-alpha coordinates found")
    chain = sorted(ca_by_chain, key=lambda ch: (-len(ca_by_chain[ch]), ch))[0]
    coords = {key: np.stack(value) for key, value in ligand_coords.items() if value}
    return chain, ca_by_chain[chain], coords


def position_features(canonical: str, cif: dict, pdb_id: str, expected_ligands: list[str]):
    entity, deposited = select_cftr_entity(cif, canonical)
    ref_to_deposited, mismatches, identity = align_to_canonical(canonical, deposited)
    chain, ca, ligands = parse_coordinates(cif, entity)
    inv = {dep: ref for ref, dep in ref_to_deposited.items()}
    coords_by_ref = {inv[label_idx]: xyz for label_idx, xyz in ca.items() if label_idx in inv}
    positions = sorted(coords_by_ref)
    pos_matrix = np.asarray(positions, dtype=int)
    xyz_matrix = np.array([coords_by_ref[p] for p in positions], dtype=float)
    contact_counts = {}
    if len(positions):
        # Reference CA contact count: exclude local +/-3 residues, 8-A spatial cutoff.
        for j, p in enumerate(positions):
            distances = np.linalg.norm(xyz_matrix - xyz_matrix[j], axis=1)
            contact_counts[p] = int(np.sum((distances < 8.0) & (np.abs(pos_matrix - p) > 3)))
    ligand_distance = {}
    missing_ligands = []
    for name in expected_ligands:
        if name not in ligands:
            missing_ligands.append(name)
            ligand_distance[name] = {}
            continue
        cloud = ligands[name]
        # CA to nearest observed ligand heavy atom, not docking energy or true contact score.
        ligand_distance[name] = {p: float(np.linalg.norm(cloud - coords_by_ref[p], axis=1).min())
                                 for p in positions}
    info = {
        "pdb_id": pdb_id, "cftr_entity_id": entity, "cftr_chain_label_asym_id": chain,
        "deposited_polymer_length": len(deposited), "mapped_residues": len(ref_to_deposited),
        "resolved_mapped_residues": len(coords_by_ref), "sequence_alignment_identity": float(identity),
        "canonical_mismatches": {str(pos): list(pair) for pos, pair in sorted(mismatches.items())},
        "expected_ligands": expected_ligands, "ligands_without_coordinates": missing_ligands,
    }
    # Explicit F508del sanity check for the known experimental construct.
    if pdb_id == "8EIQ" and 508 in ref_to_deposited:
        raise ValueError("8EIQ is annotated F508del yet maps canonical F508: inspect alignment before use")
    rows = {}
    for p in range(1, len(canonical) + 1):
        idx = ref_to_deposited.get(p)
        if idx is None:
            status = "not_in_deposited_construct"
        elif idx not in ca:
            status = "deposited_but_unresolved"
        elif p in mismatches:
            status = "resolved_construct_mismatch"
        else:
            status = "resolved_exact"
        row = {f"{pdb_id}_status": status,
               f"{pdb_id}_label_seq_id": idx,
               f"{pdb_id}_ca_contacts_8A": contact_counts.get(p),
               f"{pdb_id}_has_reference_coordinates": bool(idx in ca) if idx is not None else False}
        for ligand in expected_ligands:
            row[f"{pdb_id}_{ligand}_ca_distance_A"] = ligand_distance[ligand].get(p)
        rows[p] = row
    return rows, info


def load_identity_only(root: Path) -> tuple[pd.DataFrame, dict]:
    """Read ONLY variant identity + split membership, never paired outcome files."""
    result = []
    hashes = {}
    for study, (filename, split_col, expected) in SOURCE_META.items():
        path = root / "data" / "processed" / filename
        columns = ["variant_key", "protein_position", split_col]
        subset = pd.read_parquet(path, columns=columns)
        if len(subset) != expected or subset.variant_key.isna().any() or subset.variant_key.duplicated().any():
            raise ValueError(f"{study}: unexpected frozen identity table size/keys: {len(subset)}")
        subset["study"] = study
        subset["partition"] = subset[split_col].astype(str).str.lower()
        subset["protein_position"] = pd.to_numeric(subset["protein_position"], errors="coerce").astype("Int64")
        if subset["protein_position"].dropna().lt(1).any() or subset["protein_position"].dropna().gt(1480).any():
            raise ValueError(f"{study}: invalid protein position")
        if not set(subset.partition).issubset({"train", "val", "test", "excluded"}):
            raise ValueError(f"{study}: unrecognised frozen split label")
        result.append(subset[["study", "variant_key", "partition", "protein_position"]].copy())
        hashes[filename] = sha256_file(path)
    return pd.concat(result, ignore_index=True), hashes


def categorise_coverage(row: pd.Series, pdb_ids=("6O2P", "8EIQ")) -> str:
    if pd.isna(row["protein_position"]):
        return "unparseable_variant_position"
    supported = [pdb for pdb in pdb_ids if row[f"{pdb}_status"] == "resolved_exact"]
    if len(supported) == len(pdb_ids):
        return "exact_both"
    if len(supported) == 1:
        return f"exact_{supported[0]}_only"
    if any(row[f"{pdb}_status"] == "resolved_construct_mismatch" for pdb in pdb_ids):
        return "construct_mismatch_only_or_unresolved"
    return "no_exact_reference_coordinates"


def summarise_coverage(variants: pd.DataFrame) -> list[dict]:
    summaries = []
    for (study, partition), frame in variants.groupby(["study", "partition"], sort=True):
        known = frame.dropna(subset=["protein_position"])
        unique = known.drop_duplicates("protein_position")
        counts = Counter(known["coverage_category"])
        pcounts = Counter(unique["coverage_category"])
        summaries.append({
            "study": str(study), "partition": str(partition), "variants": int(len(frame)),
            "variants_with_position": int(len(known)),
            "distinct_positions": int(len(unique)),
            "both_exact_variant_count": int(counts["exact_both"]),
            "either_exact_variant_count": int(sum(c for k, c in counts.items() if k.startswith("exact_"))),
            "either_exact_position_count": int(sum(c for k, c in pcounts.items() if k.startswith("exact_"))),
            "either_exact_position_fraction": round(sum(c for k, c in pcounts.items() if k.startswith("exact_")) / len(unique), 6) if len(unique) else None,
            "missing_or_unparseable_variants": int(len(frame) - len(known)),
            "position_status_counts": dict(sorted(pcounts.items())),
        })
    return summaries


def run(root: Path = ROOT, download: bool = False) -> None:
    root = Path(root)
    pdb_root = root / "data/external/structures"
    print("=" * 92)
    print("CFTR-THERATYPE | PHASE 15 STRUCTURAL COVERAGE (NO OUTCOME DATA)")
    print("=" * 92)
    canonical, assay = canonical_cftr_sequence()
    print("[0] Reference: canonical UniProt P13569, 1480 aa; derived from M470 assay reference")
    print("    Canonical position 470=V, experimental assay position 470=M, F508 retained")
    position_lookup = {}
    pdb_meta = {}
    provenance = {}
    for pdb_id, expected_ligands in STRUCTURES.items():
        path = pdb_root / f"{pdb_id}.cif"
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"Missing {path}. Re-run with --download (uses Python, no curl)")
            download_reference(path, pdb_id)
        if path.stat().st_size < 100_000:
            raise ValueError(f"Suspicious structure size: {path}")
        print(f"\n[1] STRUCTURE {pdb_id} | SHA256 {sha256_file(path)}")
        cif = MMCIF2Dict(str(path))
        lookups, details = position_features(canonical, cif, pdb_id, expected_ligands)
        position_lookup[pdb_id] = lookups
        details["file_sha256"] = sha256_file(path)
        pdb_meta[pdb_id] = details
        provenance[str(path.relative_to(root))] = sha256_file(path)
        print(f"    CFTR entity={details['cftr_entity_id']}, label chain={details['cftr_chain_label_asym_id']}")
        print(f"    Deposited length={details['deposited_polymer_length']}, mapped={details['mapped_residues']}, ")
        print(f"    Resolved mapped={details['resolved_mapped_residues']}; identity={details['sequence_alignment_identity']:.4f}")
        print(f"    Construct differences: {details['canonical_mismatches']}")
        print(f"    Drug ligands: {', '.join(expected_ligands)}; missing coordinates={details['ligands_without_coordinates']}")
    identities, hashes = load_identity_only(root)
    provenance.update({f"data/processed/{name}": sha for name, sha in hashes.items()})
    positions = sorted(int(v) for v in identities.protein_position.dropna().unique())
    structural = pd.DataFrame({"protein_position": positions})
    for pdb_id in STRUCTURES:
        lookup = position_lookup[pdb_id]
        structural = structural.merge(pd.DataFrame([dict(protein_position=p, **lookup[p]) for p in positions]),
                                      on="protein_position", validate="one_to_one")
    known = identities.dropna(subset=["protein_position"]).copy()
    known["protein_position"] = known.protein_position.astype(int)
    result = known.merge(structural, on="protein_position", validate="many_to_one")
    unknown = identities.loc[identities.protein_position.isna()].copy()
    if not unknown.empty:
        result = pd.concat([result, unknown], ignore_index=True)
    result["coverage_category"] = result.apply(categorise_coverage, axis=1)
    cols = ["study", "partition", "variant_key", "protein_position", "coverage_category"]
    other_cols = [c for c in result.columns if c not in cols]
    result = result[cols + other_cols].sort_values(["study", "partition", "variant_key"]).reset_index(drop=True)
    if len(result) != len(identities) or result[["study", "variant_key"]].duplicated().any():
        raise ValueError("Lost/duplicated variants in structural coverage join")
    summaries = summarise_coverage(result)
    print("\n[2] COVERAGE PER FROZEN SPLIT (REFERENCE COORDINATES, NOT MUTANT STRUCTURES)")
    print(f"{'Study':<6} {'Split':<9} {'Variants':>8} {'Positions':>9} {'Exact either':>12} {'Covered %':>10} {'Unknown':>8}")
    for row in summaries:
        frac = row["either_exact_position_fraction"]
        print(f"{row['study']:<6} {row['partition']:<9} {row['variants']:>8} {row['distinct_positions']:>9} "
              f"{row['either_exact_position_count']:>12} {100*frac if frac is not None else 0:>9.1f}% "
              f"{row['missing_or_unparseable_variants']:>8}")
    print("\n[3] POSITION COVERAGE DETAILS (UNION OF BOTH STUDY POSITIONS)")
    for (study, partition), group in result.groupby(["study", "partition"]):
        unique = group.dropna(subset=["protein_position"]).drop_duplicates("protein_position")
        counts = Counter(unique.coverage_category)
        print(f"  {study:3s} {partition:8s}: both={counts['exact_both']:3d}, "
              f"only6O2P={counts['exact_6O2P_only']:3d}, "
              f"only8EIQ={counts['exact_8EIQ_only']:3d}, "
              f"mismatch/no-exact={len(unique) - sum(v for k, v in counts.items() if k.startswith('exact_')):3d}")
    for pdb_id in STRUCTURES:
        totals = Counter(structural[f"{pdb_id}_status"])
        print(f"  {pdb_id} coordinate-status counts: {dict(sorted(totals.items()))}")
        print(f"  {pdb_id} F508={position_lookup[pdb_id][508][f'{pdb_id}_status']}; "
              f"M470 background={position_lookup[pdb_id][470][f'{pdb_id}_status']}")
    print("  Distances are reference CA-to-experimental-ligand-atom distances (Angstrom), not binding predictions.")
    out = root / "results/structure"
    out.mkdir(parents=True, exist_ok=True)
    # Idempotent result regeneration is intentional; never modify split/structure input files.
    varfile = out / "phase15_coverage_variants.csv"
    posfile = out / "phase15_coverage_positions.csv"
    result.to_csv(varfile, index=False)
    structural.to_csv(posfile, index=False)
    report = {
        "phase": "15A_structure_coverage", "reference": "UniProt P13569 canonical (V470)",
        "assay_reference_difference": "M470 instead of canonical V470", "pdb_sources": pdb_meta,
        "file_sha256": provenance, "split_coverage": summaries,
        "total_variants": int(len(result)), "unique_positions_across_studies": len(positions),
        "outcome_data_accessed": False, "structure_is_variant_specific": False,
        "drug_binding_prediction": False, "caveats": [
            "PDB structures represent experimentally selected CFTR constructs and conformations; not every mutant",
            "8EIQ contains F508del: residue 508 cannot be mapped to its protein coordinates",
            "Study VTI uses vanzacaftor/tezacaftor pretreatment; 8EIQ contains elexacaftor/tezacaftor/ivacaftor, not vanzacaftor",
            "Computed CA-ligand distances are reference-structure geometry only; no binding/efficacy inference",
            "Frozen split identity tables read; no paired assay/label outcome table opened",
        ],
    }
    summaryfile = out / "phase15_coverage_summary.json"
    summaryfile.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("\n[4] SAVED")
    for path in (varfile, posfile, summaryfile):
        print(f"  {path.relative_to(root)}")
    print("PASS: No outcomes, validation/test measurements, or mutant structures accessed")
    print("=" * 92)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download only missing RCSB mmCIF structures using Python")
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root (for tests)")
    args = parser.parse_args()
    run(args.root, download=args.download)


if __name__ == "__main__":
    main()
