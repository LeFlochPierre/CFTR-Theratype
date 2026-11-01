from pathlib import Path
import csv
import statistics as stats
import numpy as np
import gemmi

ROOT = Path.home() / "projects/CFTR-Theratype"
BASE = ROOT / "results/boltz/phase18c"
REFERENCE = ROOT / "data/external/structures/6O2P.cif"
SEEDS = [11, 22, 33]

VARIANTS = {
    "Ser308Pro":  ("seeded",   "mut_pSer308Pro", 308),
    "Gly314Val":  ("expanded", "mut_pGly314Val", 314),
    "Arg352Gly":  ("expanded", "mut_pArg352Gly", 352),
    "Tyr917Cys":  ("expanded", "mut_pTyr917Cys", 917),
    "Asp1445Asn": ("seeded",   "mut_pAsp1445Asn", 1445),
    "Leu633Ile":  ("expanded", "mut_pLeu633Ile", 633),
}

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

def load(path):
    table = gemmi.cif.read_file(str(path)).sole_block().find(TAGS)
    assert len(table), f"No atom records: {path}"

    ca, atoms, ligand = {}, {}, {}

    for group, chain, seq, res, atom, element, xs, ys, zs in table:
        if element.upper() in ("H", "D"):
            continue

        xyz = np.array(
            [float(xs), float(ys), float(zs)],
            dtype=float,
        )

        if group == "ATOM" and chain == "A" and seq not in (".", "?"):
            pos = int(seq)
            atoms[(pos, atom)] = (res, xyz)
            if atom == "CA":
                ca[pos] = (res, xyz)

        if group == "HETATM" and res == "VX7":
            ligand[atom] = xyz

    assert len(ca) >= 700, f"Insufficient protein atoms: {path}"
    assert len(ligand) >= 25, f"VX7 incomplete: {path}"

    return {"ca": ca, "atoms": atoms, "ligand": ligand}

def get_prediction(run):
    paths = sorted(run.rglob("*_model_*.cif"))
    assert len(paths) == 1, (run, len(paths))
    return load(paths[0])

def fit(moving, fixed):
    """Rigidly align moving coordinates to fixed coordinates."""
    mx, fx = moving.mean(0), fixed.mean(0)

    u, _, vt = np.linalg.svd(
        (moving - mx).T @ (fixed - fx)
    )

    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))

    rotation = u @ correction @ vt
    translation = fx - mx @ rotation

    aligned = moving @ rotation + translation

    rmsd = float(np.sqrt(np.mean(
        np.sum((aligned - fixed) ** 2, axis=1)
    )))

    return rotation, translation, rmsd

def contacts(data, cutoff=4.5):
    ligand = np.array(list(data["ligand"].values()))
    found = set()

    for (pos, atom), (_, xyz) in data["atoms"].items():
        if np.min(np.linalg.norm(ligand - xyz, axis=1)) <= cutoff:
            found.add(pos)

    return found

def mutation_distance(data, pos):
    ligand = np.array(list(data["ligand"].values()))
    site = np.array([
        xyz
        for (p, _), (_, xyz) in data["atoms"].items()
        if p == pos
    ])

    assert len(site), f"Mutation site {pos} missing"

    return float(np.min(np.linalg.norm(
        ligand[:, None, :] - site[None, :, :],
        axis=2
    )))

reference = load(REFERENCE)
reference_ligand = np.array(list(reference["ligand"].values()))

# Same experimentally defined 15 Å pocket as the earlier validator.
pocket = sorted(
    p for p, (_, xyz) in reference["ca"].items()
    if np.min(np.linalg.norm(reference_ligand - xyz, axis=1)) <= 15
)

print("=" * 105)
print("PHASE 18E | DIRECT MUTANT-TO-MATCHED-WT STRUCTURAL COMPARISON")
print("=" * 105)
print(f"Reference pocket Cα positions: {len(pocket)}")
print("Alignment: same-seed predicted WT, excluding mutation site")
print("Ligand: identical named VX7 heavy atoms")
print("Contact cutoff: 4.5 Å")

rows = []

for seed in SEEDS:
    wt_run = BASE / "seeded" / f"seed_{seed}" / "wt_6O2P_VX7_guided"
    wt = get_prediction(wt_run)
    wt_contacts = contacts(wt)

    for name, (folder, filename, mut_pos) in VARIANTS.items():
        run = BASE / folder / f"seed_{seed}" / filename
        mutant = get_prediction(run)
        mutant_contacts = contacts(mutant)

        # Match equivalent pocket backbone atoms.
        # Exclude the substituted position.
        positions = [
            p for p in pocket
            if p != mut_pos
            and p in wt["ca"]
            and p in mutant["ca"]
            and wt["ca"][p][0] == mutant["ca"][p][0]
        ]

        assert len(positions) >= 10, (seed, name, len(positions))

        moving = np.array([mutant["ca"][p][1] for p in positions])
        fixed = np.array([wt["ca"][p][1] for p in positions])

        rotation, translation, pocket_ca_rmsd = fit(moving, fixed)

        # Direct ligand coordinates after local protein alignment.
        ligand_atoms = sorted(
            wt["ligand"].keys() & mutant["ligand"].keys()
        )
        assert len(ligand_atoms) >= 25

        wt_ligand = np.array([
            wt["ligand"][a] for a in ligand_atoms
        ])
        mut_ligand = np.array([
            mutant["ligand"][a] for a in ligand_atoms
        ]) @ rotation + translation

        ligand_rmsd = float(np.sqrt(np.mean(
            np.sum((mut_ligand - wt_ligand) ** 2, axis=1)
        )))

        centroid = float(np.linalg.norm(
            mut_ligand.mean(0) - wt_ligand.mean(0)
        ))

        # Side-chain changes in the pocket, excluding mutation site.
        # Compare identical atom names in identical residue types.
        backbone = {"N", "CA", "C", "O", "OXT"}
        squared = []

        for key, (wt_res, wt_xyz) in wt["atoms"].items():
            p, atom = key
            if p not in positions or atom in backbone:
                continue
            if key not in mutant["atoms"]:
                continue

            mut_res, mut_xyz = mutant["atoms"][key]
            if wt_res != mut_res:
                continue

            aligned = mut_xyz @ rotation + translation
            squared.append(float(np.sum((aligned - wt_xyz) ** 2)))

        sidechain_rmsd = (
            float(np.sqrt(np.mean(squared))) if squared else float("nan")
        )

        union = wt_contacts | mutant_contacts
        overlap = len(wt_contacts & mutant_contacts) / len(union)
        lost = sorted(wt_contacts - mutant_contacts)
        gained = sorted(mutant_contacts - wt_contacts)

        rows.append({
            "seed": seed,
            "variant": name,
            "mutation_position": mut_pos,
            "alignment_ca_count": len(positions),
            "ligand_direct_rmsd_A": ligand_rmsd,
            "ligand_centroid_shift_A": centroid,
            "pocket_ca_rmsd_A": pocket_ca_rmsd,
            "pocket_sidechain_rmsd_A": sidechain_rmsd,
            "contact_jaccard_vs_wt": overlap,
            "lost_contact_positions": ";".join(map(str, lost)),
            "gained_contact_positions": ";".join(map(str, gained)),
            "wt_site_ligand_min_A": mutation_distance(wt, mut_pos),
            "mut_site_ligand_min_A": mutation_distance(mutant, mut_pos),
        })

assert len(rows) == 18

output = BASE / "direct_wt_comparison.csv"

with output.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

print("\n" + "=" * 105)
print("PER-SEED DIRECT STRUCTURAL DIFFERENCES")
print("=" * 105)

print(
    f"{'Seed':<6} {'Variant':<15} {'Lig RMSD':>10} "
    f"{'Centroid':>10} {'Pocket CA':>11} "
    f"{'Sidechain':>11} {'Contacts':>10}"
)

for r in rows:
    print(
        f"{r['seed']:<6} {r['variant']:<15} "
        f"{r['ligand_direct_rmsd_A']:>10.3f} "
        f"{r['ligand_centroid_shift_A']:>10.3f} "
        f"{r['pocket_ca_rmsd_A']:>11.3f} "
        f"{r['pocket_sidechain_rmsd_A']:>11.3f} "
        f"{r['contact_jaccard_vs_wt']:>10.3f}"
    )

print("\n" + "=" * 105)
print("MUTATION SUMMARY | MEAN ± SD ACROSS THREE MATCHED SEEDS")
print("=" * 105)

print(
    f"{'Variant':<15} {'Ligand RMSD Å':>19} "
    f"{'Pocket CA Å':>18} {'Contacts':>17}"
)

for name in VARIANTS:
    selected = [r for r in rows if r["variant"] == name]

    def summary(key):
        values = [r[key] for r in selected]
        return f"{stats.mean(values):.3f} ± {stats.stdev(values):.3f}"

    print(
        f"{name:<15} "
        f"{summary('ligand_direct_rmsd_A'):>19} "
        f"{summary('pocket_ca_rmsd_A'):>18} "
        f"{summary('contact_jaccard_vs_wt'):>17}"
    )

print("\n" + "=" * 105)
print("CONTACT-NETWORK CHANGES")
print("=" * 105)

for r in rows:
    print(
        f"{r['variant']:<15} seed={r['seed']} "
        f"lost=[{r['lost_contact_positions']}] "
        f"gained=[{r['gained_contact_positions']}]"
    )

print("\nQC: 18/18 matched mutant–WT comparisons")
print("Saved:", output)
print("PASS: Phase 18E complete")
