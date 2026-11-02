"""Phase 18C: label-free, construct-mapped CFTR/ivacaftor Boltz-2 mutant pilot.

CLI: python -m cftr_theratype.boltz_pilot prepare

Reads ONLY Phase 16A frozen VTI train identity metadata and Phase 15A experimental
6O2P reference information. Does NOT read ETI/VTI treatment outcomes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from cftr_theratype.structure_coverage import (
    ROOT, align_to_canonical, canonical_cftr_sequence, parse_coordinates,
    select_cftr_entity, sha256_file,
)

EXPECTED_COUNTS = {"eligible": 195}
WORK = Path("results/boltz/phase18c")
SRC_YAML = Path("results/boltz/phase18b/inputs/6O2P_ivacaftor_pocket_guided.yaml")
CANDIDATES = Path("results/structure/phase16_foldx/candidates_train_only.csv")
FROZEN_MANIFEST = Path("results/structure/phase16_foldx/manifest.json")
CIF = Path("data/external/structures/6O2P.cif")
HGVS = re.compile(r"^p\.\(?([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})\)?$")
AA3 = dict(zip(
    ["Ala", "Cys", "Asp", "Glu", "Phe", "Gly", "His", "Ile", "Lys", "Leu", "Met", "Asn", "Pro", "Gln", "Arg", "Ser", "Thr", "Val", "Trp", "Tyr"],
    "ACDEFGHIKLMNPQRSTVWY",
))


def parse_hgvs(value):
    match = HGVS.fullmatch(str(value).strip())
    if match is None:
        raise ValueError(f"Unsupported missense HGVS: {value!r}")
    a, pos, b = match.groups()
    if a not in AA3 or b not in AA3:
        raise ValueError(f"Unsupported amino acid: {value}")
    return AA3[a], int(pos), AA3[b]


def mutate_reference(base_sequence: str, deposited_pos: int, wt: str, mutant: str):
    """Enforce a single, known-residue substitution in deposited polymer indexing."""
    if not (1 <= deposited_pos <= len(base_sequence)):
        raise ValueError("Deposited position out of range")
    if base_sequence[deposited_pos - 1] != wt or mutant == wt:
        raise ValueError(f"Reference mismatch at {deposited_pos}: expected {wt}, "
                         f"found {base_sequence[deposited_pos - 1]}")
    result = base_sequence[:deposited_pos-1] + mutant + base_sequence[deposited_pos:]
    if len(result) != len(base_sequence):
        raise AssertionError("Mutated sequence length changed")
    differences = [i + 1 for i, (a, b) in enumerate(zip(base_sequence, result)) if a != b]
    if differences != [deposited_pos]:
        raise AssertionError(f"Mutation changed unexpected sites: {differences}")
    return result


def select_panel(records: list[dict], n_near: int = 4, n_mid: int = 4, n_far: int = 4):
    """Deterministic, distance-only site selection; no response labels involved."""
    eligible = sorted(records, key=lambda r: (r["distance_to_vx7_A"], r["protein_position"], str(r["variant_key"])))
    unique, seen = [], set()
    for r in eligible:
        if r["protein_position"] not in seen:
            seen.add(r["protein_position"])
            unique.append(dict(r))
    if len(unique) < n_near + n_mid + n_far:
        raise ValueError("Insufficient unique structurally resolved positions")
    near = unique[:n_near]
    far = unique[-n_far:][::-1]
    reserved = {r["protein_position"] for r in near + far}
    middle = [r for r in unique if r["protein_position"] not in reserved]
    indices = np.floor((np.arange(n_mid) + 0.5) * len(middle) / n_mid).astype(int)
    mid = [middle[i] for i in indices]
    panel = [("near", r) for r in near] + [("intermediate", r) for r in mid] + [("distant", r) for r in far]
    if len({r["protein_position"] for _, r in panel}) != len(panel):
        raise AssertionError("Pilot sites must be unique")
    return panel


def prepare(root: Path = ROOT):
    import yaml  # PyYAML; available in BOLTZ2_ENV / install in CFTR_ENV if necessary
    root = Path(root).resolve()
    cif_path = root / CIF
    source_yaml = root / SRC_YAML
    candidates_path = root / CANDIDATES
    manifest_path = root / FROZEN_MANIFEST
    for path in [cif_path, source_yaml, candidates_path, manifest_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Required input not found: {path}")

    manifest = json.loads(manifest_path.read_text())
    assert manifest["source_cif_sha256"] == sha256_file(cif_path), "Source CIF differs from frozen manifest"
    assert manifest["prepared_pdb_sha256"] == sha256_file(
        root / "results/structure/phase16_foldx/6O2P_cftr_canonical_numbering.pdb"
    ), "Reference protein structure differs from frozen manifest"

    cif = MMCIF2Dict(str(cif_path))
    canonical, assay = canonical_cftr_sequence()
    cftr_entity, deposited = select_cftr_entity(cif, canonical)
    mapping, mismatches, identity = align_to_canonical(canonical, deposited)
    assert identity > 0.99 and len(mapping) >= 1480
    assert mismatches == {1371: ("E", "Q")}, f"Unexpected construct differences: {mismatches}"
    chain, ca, ligands = parse_coordinates(cif, cftr_entity)
    assert chain == "A", f"Unexpected CFTR polymer chain: {chain}"
    assert len(ca) == 1179, f"Unexpected number of resolved CA residues: {len(ca)}"
    vx7 = ligands.get("VX7")
    assert vx7 is not None and len(vx7) == 29, "Expected 29 VX7 heavy atoms"

    cfg = yaml.safe_load(source_yaml.read_text())
    protein = [x["protein"] for x in cfg["sequences"] if "protein" in x]
    ligand = [x["ligand"] for x in cfg["sequences"] if "ligand" in x]
    assert len(protein) == 1 and len(ligand) == 1
    assert protein[0]["id"] == "A" and ligand[0]["id"] == "B" and ligand[0]["ccd"] == "VX7"
    assert protein[0]["sequence"] == deposited, "YAML sequence differs from experimental 6O2P polymer"
    assert len(deposited) == 1489, "Unexpected 6O2P construct length"
    assert cfg["templates"] == [{"cif": str(cif_path), "chain_id": "A"}], "Template changed"
    assert cfg["properties"] == [{"affinity": {"binder": "B"}}], "Affinity property changed"
    constraints = cfg["constraints"]
    assert len(constraints) == 1 and "pocket" in constraints[0], "Expected exactly one guided pocket constraint"
    pocket = constraints[0]["pocket"]
    assert pocket == {
        "binder": "B", "contacts": [["A", k] for k in [233, 305, 312, 930, 931]],
        "max_distance": 6.0, "force": True,
    }, "Pocket guidance differs from validated WT run"

    with candidates_path.open(newline="") as f:
        candidates = list(csv.DictReader(f))
    assert len(candidates) == 210, f"Expected 210 VTI train identities; got {len(candidates)}"
    assert sum(r["status"] == "eligible" for r in candidates) == 195, "Expected 195 eligible VTI training variants"
    assert len({r["variant_key"] for r in candidates}) == 210
    if manifest.get("n_train") is not None:
        assert manifest["n_train"] == 210

    records = []
    for r in candidates:
        if r["status"] != "eligible":
            continue
        wt, pos, mut = parse_hgvs(r["hgvs_protein"])
        assert int(r["protein_position"]) == pos
        assert 1 <= pos <= len(canonical) and wt == canonical[pos-1] and wt == assay[pos-1]
        assert pos not in (470,1371)
        deposited_pos = mapping[pos]
        assert deposited[deposited_pos - 1] == wt, (r["hgvs_protein"], deposited_pos)
        assert deposited_pos in ca, f"Eligible position absent from mmCIF CA: {r['hgvs_protein']}"
        distance = float(np.min(np.linalg.norm(vx7 - ca[deposited_pos], axis=1)))
        records.append({
            "variant_key": r["variant_key"], "hgvs_protein": r["hgvs_protein"],
            "protein_position": pos, "deposited_sequence_position": deposited_pos,
            "reference_aa": wt, "alternate_aa": mut,
            "distance_to_vx7_A": round(distance, 4),
        })
    assert len(records) == 195

    panel = select_panel(records)
    output = root / WORK
    input_dir = output / "inputs_full"
    smoke_dir = output / "inputs_smoke"
    if input_dir.exists() or smoke_dir.exists():
        raise FileExistsError("Existing Phase 18C inputs detected. Refusing to overwrite; archive or move them first.")
    input_dir.mkdir(parents=True)
    smoke_dir.mkdir(parents=True)

    def write_case(filename: str, new_sequence: str):
        from copy import deepcopy
        conf = deepcopy(cfg)
        next(x["protein"] for x in conf["sequences"] if "protein" in x)["sequence"] = new_sequence
        path = input_dir / filename
        path.write_text(yaml.safe_dump(conf, sort_keys=False))
        return path

    wt_path = write_case("wt_6O2P_VX7_guided.yaml", deposited)
    shutil.copy2(wt_path, smoke_dir / wt_path.name)
    rows = []
    for cohort, r in panel:
        seq = mutate_reference(
            deposited, r["deposited_sequence_position"], r["reference_aa"], r["alternate_aa"]
        )
        file = write_case("mut_" + r["hgvs_protein"].replace("p.", "p").replace(".", "") + ".yaml", seq)
        rows.append({**r, "structural_group": cohort, "input_file": str(file.relative_to(root))})
    # Smoke test: WT, one pocket-proximal mutation, one distant negative control.
    for record in [rows[0], rows[-1]]:
        candidate = root / record["input_file"]
        shutil.copy2(candidate, smoke_dir / candidate.name)

    with (output / "pilot_train_only.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "protocol": "phase18c_label_free_6O2P_VX7_pocket_guided_pilot_v1",
        "reference_cif_sha256": sha256_file(cif_path),
        "source_yaml_sha256": sha256_file(source_yaml),
        "training_candidate_manifest_sha256": sha256_file(candidates_path),
        "reference_construct_length": len(deposited),
        "canonical_length": len(canonical),
        "mapped_canonical_positions": len(mapping),
        "resolved_ca_positions": len(ca),
        "eligible_train_variants": len(records),
        "pilot_variants": len(rows),
        "wt_controls": 1,
        "smoke_cases": 3,
        "pocket_contacts_in_6O2P_sequence_index": [233,305,312,930,931],
        "treatment_outcomes_accessed": False,
        "validation_test_outcomes_accessed": False,
        "important_caveat": "All variants use the same ligand-bound WT 6O2P template and forced pocket; this is a constrained pose perturbation experiment, not a thermodynamic free energy calculation.",
        "selected_panel": rows,
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print("=" * 85)
    print("PHASE 18C | LABEL-FREE CFTR–IVACAFTOR MUTATION PILOT")
    print("=" * 85)
    print(f"Reference: experimental 6O2P / {len(deposited)} aa / {len(ca)} resolved CA")
    print(f"Canonical-to-deposited alignment: {len(mapping)} mapped residues / identity {identity:.5f}")
    print("Background differences: canonical V470, engineered E1371Q; assay M470")
    print(f"VTI training candidates: 210 / structurally eligible: {len(records)} / pilot: {len(panel)}")
    print(f"{'Category':15s} {'Mutation':18s} {'Canonical':>10s} {'6O2P':>7s} {'VX7 Å':>9s}")
    for category, r in panel:
        print(f"{category:15s} {r['hgvs_protein']:18s} {r['protein_position']:10d} "
              f"{r['deposited_sequence_position']:7d} {r['distance_to_vx7_A']:9.2f}")
    print("Outputs:", output)
    print("Smoke inputs:", smoke_dir, "(WT + nearest mutation + farthest control)")
    print("PASS: exactly one amino-acid change per mutant; no response data accessed")
    print("=" * 85)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare"])
    p.add_argument("--root", type=Path, default=ROOT)
    args = p.parse_args()
    if args.action == "prepare":
        prepare(args.root)


if __name__ == "__main__":
    main()
