from pathlib import Path
from collections import Counter
import json
import numpy as np
import gemmi

ROOT = Path.home() / "projects/CFTR-Theratype"
REFERENCE = ROOT / "data/external/structures/6O2P.cif"
RUN = ROOT / "results/boltz/phase18b/run_native_cif"
OUTPUT = ROOT / "results/boltz/phase18b/pose_validation.json"

predictions = sorted(RUN.rglob("*_model_*.cif"))
assert len(predictions) == 1, (
    f"Expected exactly one predicted structure; found {len(predictions)}"
)
PREDICTED = predictions[0]

TAGS = [
    "_atom_site.group_PDB",
    "_atom_site.label_asym_id",
    "_atom_site.label_seq_id",
    "_atom_site.label_comp_id",
    "_atom_site.label_atom_id",
    "_atom_site.type_symbol",
    "_atom_site.Cartn_x",
    "_atom_site.Cartn_y",
    "_atom_site.Cartn_z",
]

def load_structure(path):
    block = gemmi.cif.read_file(str(path)).sole_block()
    table = block.find(TAGS)
    assert len(table) > 0, f"No atom records: {path}"

    ca, ligand, protein = {}, {}, []
    hetero = Counter()

    for row in table:
        group, chain, seq, res, atom, element, xs, ys, zs = row

        xyz = np.array(
            [float(xs), float(ys), float(zs)],
            dtype=float
        )

        if group == "HETATM":
            hetero[res] += 1

        if element.upper() in ("H", "D"):
            continue

        if group == "ATOM" and chain == "A" and seq not in (".", "?"):
            pos = int(seq)
            protein.append((pos, xyz))
            if atom == "CA":
                ca[pos] = (res, xyz)

        if group == "HETATM" and res == "VX7":
            ligand[atom] = xyz

    return {
        "ca": ca,
        "ligand": ligand,
        "protein": protein,
        "hetero": hetero,
    }

ref = load_structure(REFERENCE)
pred = load_structure(PREDICTED)

for name, data in [("Reference", ref), ("Prediction", pred)]:
    print(
        f"{name}: {len(data['ca'])} CA residues, "
        f"{len(data['ligand'])} VX7 heavy atoms"
    )
    if not data["ligand"]:
        print("Other HETATM components:", data["hetero"].most_common(12))
        raise SystemExit(f"FAIL: VX7 not identified in {name}")

# Align residues by deposited polymer sequence position, not author numbering.
common_ca = sorted(
    p for p in ref["ca"].keys() & pred["ca"].keys()
    if ref["ca"][p][0] == pred["ca"][p][0]
)

assert len(common_ca) >= 700, (
    f"Insufficient matching CFTR residues: {len(common_ca)}. "
    "Check sequence/chain mapping before comparing coordinates."
)

# Only compare equivalent, named heavy atoms of VX7.
common_atoms = sorted(ref["ligand"].keys() & pred["ligand"].keys())
assert len(common_atoms) >= 25, (
    f"Insufficient matching ligand atoms: {len(common_atoms)}. "
    "Verify VX7 atom naming."
)

def superpose(positions):
    x = np.array([pred["ca"][p][1] for p in positions])
    y = np.array([ref["ca"][p][1] for p in positions])

    xc, yc = x.mean(axis=0), y.mean(axis=0)
    u, _, vt = np.linalg.svd((x - xc).T @ (y - yc))

    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))

    rotation = u @ correction @ vt
    translation = yc - xc @ rotation

    fitted = x @ rotation + translation
    protein_rmsd = np.sqrt(np.mean(np.sum((fitted - y)**2, axis=1)))

    return rotation, translation, float(protein_rmsd)

experimental_ligand = np.array(
    [ref["ligand"][a] for a in common_atoms]
)
predicted_ligand = np.array(
    [pred["ligand"][a] for a in common_atoms]
)

# Global CFTR alignment.
r_global, t_global, global_protein_rmsd = superpose(common_ca)
aligned_global = predicted_ligand @ r_global + t_global

global_ligand_rmsd = float(np.sqrt(
    np.mean(np.sum(
        (aligned_global - experimental_ligand)**2, axis=1
    ))
))

# Local pocket alignment, to distinguish docking errors from
# larger-scale differences in CFTR conformation.
pocket_ca = [
    p for p in common_ca
    if np.min(np.linalg.norm(
        experimental_ligand - ref["ca"][p][1], axis=1
    )) <= 15.0
]

assert len(pocket_ca) >= 10, (
    f"Only {len(pocket_ca)} pocket CA residues. "
    "Check ligand placement and residue mapping."
)

r_local, t_local, local_protein_rmsd = superpose(pocket_ca)
aligned_local = predicted_ligand @ r_local + t_local

local_ligand_rmsd = float(np.sqrt(
    np.mean(np.sum(
        (aligned_local - experimental_ligand)**2, axis=1
    ))
))

centroid_error = float(np.linalg.norm(
    aligned_local.mean(axis=0) - experimental_ligand.mean(axis=0)
))

def contact_residues(structure, cutoff=4.5):
    ligand_xyz = np.array(list(structure["ligand"].values()))
    contacts = set()
    for pos, xyz in structure["protein"]:
        if np.min(np.linalg.norm(ligand_xyz - xyz, axis=1)) <= cutoff:
            contacts.add(pos)
    return contacts

ref_contacts = contact_residues(ref)
pred_contacts = contact_residues(pred)
intersection = ref_contacts & pred_contacts
union = ref_contacts | pred_contacts
jaccard = len(intersection) / len(union) if union else None

results = {
    "reference": str(REFERENCE),
    "prediction": str(PREDICTED),
    "matched_protein_ca": len(common_ca),
    "matched_ligand_heavy_atoms": len(common_atoms),
    "pocket_alignment_ca": len(pocket_ca),
    "global_protein_rmsd_A": global_protein_rmsd,
    "local_protein_rmsd_A": local_protein_rmsd,
    "global_ligand_rmsd_A": global_ligand_rmsd,
    "local_ligand_rmsd_A": local_ligand_rmsd,
    "local_ligand_centroid_error_A": centroid_error,
    "reference_contact_positions": sorted(ref_contacts),
    "predicted_contact_positions": sorted(pred_contacts),
    "contact_jaccard": jaccard,
    "note": "Residue indices are 6O2P construct positions, not canonical CFTR numbering. Ligand RMSD uses matching CCD atom names, without symmetry correction.",
}

OUTPUT.write_text(json.dumps(results, indent=2))

print("\n" + "=" * 68)
print("PHASE 18B | CFTR–IVACAFTOR BINDING POSE VALIDATION")
print("=" * 68)

for key in [
    "matched_protein_ca",
    "matched_ligand_heavy_atoms",
    "global_protein_rmsd_A",
    "local_protein_rmsd_A",
    "global_ligand_rmsd_A",
    "local_ligand_rmsd_A",
    "local_ligand_centroid_error_A",
    "contact_jaccard",
]:
    print(f"{key}: {results[key]}")

print("Reference contact positions:", sorted(ref_contacts))
print("Predicted contact positions:", sorted(pred_contacts))
print("Shared contact positions:", sorted(intersection))
print("Saved:", OUTPUT)
