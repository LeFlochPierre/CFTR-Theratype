"""Synthetic local PDB provenance parity for the Phase27D3A/28C refactor."""
from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path

import pandas as pd
import pytest

from cftr_theratype.features.structure_provenance import (
    SPECS,
    verify_provenance,
)


ROOT = Path(__file__).resolve().parents[1]


def _build_provenance(tmp_path):
    root = tmp_path / "research"
    folder = root / "data/external/cftr_structures"
    folder.mkdir(parents=True)
    rows = []
    for pdb in SPECS:
        path = folder / f"{pdb}.pdb"
        path.write_bytes(f"synthetic structural coordinates for {pdb}\n".encode())
        rows.append({
            "pdb": pdb,
            "canonical_agreement": 0.999,
            "mismatches_canonical": 1,
            "first_mismatches": "(1371, Q, E)",
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    provenance = root / "results/biology/phase27d/phase27d3_structure_provenance.csv"
    provenance.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(provenance, index=False)
    return root, provenance


def test_canonical_and_legacy_validate_same_local_provenance(tmp_path, monkeypatch):
    root, provenance = _build_provenance(tmp_path)
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import phase27d3b_structure_ablation as historical

    assert historical.SPECS is SPECS
    monkeypatch.setattr(historical, "ROOT", root)
    monkeypatch.setattr(historical, "PROVENANCE", provenance)
    current = verify_provenance(provenance, root, SPECS)
    previous_api = historical.verify_provenance()
    pd.testing.assert_frame_equal(current, previous_api, check_exact=True)
    assert set(current.pdb) == {"8EIG", "7SV7"}


@pytest.mark.parametrize("failure,match", [
    ("duplicate_pdb", "Unexpected PDB provenance"),
    ("agreement", "Insufficient sequence agreement"),
    ("mismatches", "Unexpected mismatch set"),
    ("residue1371", "Unexpected mismatch set"),
    ("hash", "Structure file missing or changed"),
    ("missing_pdb", "Structure file missing or changed"),
])
def test_strict_provenance_guards(tmp_path, failure, match):
    root, provenance = _build_provenance(tmp_path)
    frame = pd.read_csv(provenance)
    if failure == "duplicate_pdb":
        frame.loc[1, "pdb"] = frame.loc[0, "pdb"]
    elif failure == "agreement":
        frame.loc[0, "canonical_agreement"] = 0.98
    elif failure == "mismatches":
        frame.loc[0, "mismatches_canonical"] = 2
    elif failure == "residue1371":
        frame.loc[0, "first_mismatches"] = "(508, F, -)"
    elif failure == "hash":
        frame.loc[0, "file_sha256"] = "0" * 64
    elif failure == "missing_pdb":
        (root / "data/external/cftr_structures/8EIG.pdb").unlink()
    frame.to_csv(provenance, index=False)
    with pytest.raises(ValueError, match=match):
        verify_provenance(provenance, root)


def test_missing_provenance_fails_without_network_or_patient_data(tmp_path):
    root = tmp_path / "repo"
    with pytest.raises(FileNotFoundError, match="Missing Phase 27D-3A"):
        verify_provenance(root / "absent.csv", root)


def test_phase28c_no_longer_imports_numbered_modules_or_scripts_path():
    path = ROOT / "scripts/phase28c_structure_biophysics.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(item.name for item in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any(module.startswith("phase") for module in imports)
    assert "cftr_theratype.features.structure_provenance" in imports
    assert not any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Attribute)
        and isinstance(n.func.value.value, ast.Name)
        and n.func.value.value.id == "sys"
        for n in ast.walk(tree)
    )
