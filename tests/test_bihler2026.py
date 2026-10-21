"""Tests for safe VTI pairing and label semantics."""

import pandas as pd
import pytest

from cftr_theratype.ingest import bihler2026 as mod


@pytest.mark.parametrize("text,expected", [
    ("DMSO", "DMSO"),
    ("VNZ + TEZ", "VNZ/TEZ"),
    ("VNZ/TEZ", "VNZ/TEZ"),
    ("vanzacaftor + tezacaftor", "VNZ/TEZ"),
    ("ELX + TEZ", "UNKNOWN"),
])
def test_treatment_name(text, expected):
    assert mod.treatment_name(text) == expected


def test_2026_build_preserves_controls_and_response(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    source = tmp_path / "mmc1.xlsx"
    output = tmp_path / "processed"
    manifest = tmp_path / "manifest.json"

    def row(variant, cdna, condition, fsk, fsk_iva):
        return {
            "Study": "VTI", "Variant": variant,
            "Response group": ">10", "HGVS cDNA": cdna,
            "HGVS protein": "p.Glu54Gly", "Variant type": "Missense",
            "Treatment": condition, "FSK, %WT": fsk,
            "FSK + IVA, %WT": fsk_iva, "# Wells": 4,
        }

    frame = pd.DataFrame([
        row("E54G", "c.161A>G", "DMSO", 20., 30.),
        row("E54G", "c.161A>G", "VNZ + TEZ", 40., 60.),
        row("G551D", "c.1652G>A", "DMSO", 5., 15.),
        row("F508DEL_ctrl", "c.1521_1523del", "DMSO", 1., 3.),
    ])
    with pd.ExcelWriter(source) as writer:
        frame.to_excel(writer, sheet_name="By name", startrow=3, index=False)

    monkeypatch.setattr(mod, "SOURCE", source)
    monkeypatch.setattr(mod, "OUTPUT", output)
    monkeypatch.setattr(mod, "MANIFEST", manifest)
    mod.build_2026()

    paired = pd.read_parquet(output / "bihler_2026_vti.parquet")
    all_conditions = pd.read_parquet(output / "bihler_2026_conditions.parquet")
    assert len(paired) == 1
    assert len(all_conditions) == 4
    assert paired.iloc[0]["variant_key"] == "c.161A>G"
    assert paired.iloc[0]["vti_rescue_pp"] == pytest.approx(40.0)
    assert paired.iloc[0]["vti_increment_over_iva_pp"] == pytest.approx(30.0)
    assert paired.iloc[0]["wells_vnz_tez"] == 4
    excluded = pd.read_csv(output / "bihler_2026_unresolved.csv")
    assert "c.1652G>A" in excluded.variant_key.tolist()
