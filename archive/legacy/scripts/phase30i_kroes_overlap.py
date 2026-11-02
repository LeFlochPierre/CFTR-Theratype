#!/usr/bin/env python3
"""Phase 30I — External organoid genotype overlap.
No model fitting or VAL/TEST assay outcomes.
Published examples are NOT the full Kroes cohort.
"""

from pathlib import Path
import re
import json
import hashlib
import pandas as pd

ROOT = Path.cwd()
OUT = ROOT / "results/biology/phase30i"

MANIFEST = ROOT / (
    "results/biology/phase28e/"
    "phase28e1_foldx_mutation_manifest.csv"
)

ETI = ROOT / (
    "results/biology/phase30c/"
    "phase30c_matched_train.csv"
)

AA3 = dict(zip(
    "Ala Arg Asn Asp Cys Gln Glu Gly His Ile "
    "Leu Lys Met Phe Pro Ser Thr Trp Tyr Val Ter".split(),
    "ARNDCQEGHILKMFPSTWYV*"
))

# Non-exhaustive examples verified from Kroes et al. 2026.
# No numerical FIS or SLA outcomes are inferred.
CASES = [
    (
        "p.Val520Phe/c.489+1G>T",
        "Reported greater VTD swelling"
    ),
    (
        "p.Gly542Ter/c.490-2A>C",
        "Reported greater VTD swelling"
    ),
    (
        "p.Gly85Glu/p.Gly542Ter",
        "Reported greater VTD swelling"
    ),
    (
        "p.Ile502Thr/p.Asn1303Lys",
        "Reached reference response with VTD only"
    ),
    (
        "p.Ala561Glu/p.Ala561Glu",
        "VTD-only SLA threshold"
    ),
    (
        "p.Arg1066Cys/p.Asn1303Lys",
        "VTD-only SLA threshold"
    ),
    (
        "p.Gly542Ter/p.Arg1066His",
        "High SLA with either treatment"
    ),
    (
        "p.Gly542Ter/p.Arg1066Cys",
        "High SLA with either treatment"
    ),
    (
        "p.Glu92Lys/p.Gln890Ter",
        "Comparable response to both treatments"
    ),
    (
        "p.Leu1077Pro/p.Trp1282Ter",
        "Comparable response to both treatments"
    ),
    (
        "p.Arg153Ile/c.3659delC",
        "ETI-only SLA threshold"
    ),
    (
        "p.Asn1303Lys/c.3873+2T>C",
        "ETI-only SLA threshold"
    ),
]


def protein_code(value):
    match = re.fullmatch(
        r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|\*)",
        value
    )

    if match is None:
        return None, "nonprotein_or_unresolved"

    wild, position, mutant = match.groups()

    if wild not in AA3 or mutant not in AA3:
        raise ValueError("Invalid amino acid: " + value)

    code = AA3[wild] + position + AA3[mutant]

    if AA3[mutant] == "*":
        return code, "nonsense"

    if wild == mutant:
        return code, "synonymous"

    return code, "missense"


def main():
    if not MANIFEST.exists():
        raise FileNotFoundError(MANIFEST)

    m = pd.read_csv(
        MANIFEST,
        dtype={"variant_key": str}
    )

    required = {
        "pdb", "variant_key",
        "mutation", "protein_position"
    }

    if not required.issubset(m.columns):
        raise ValueError("FoldX manifest schema changed")

    m = m.loc[
        m.pdb.astype(str).str.upper() == "7SV7",
        ["variant_key", "mutation", "protein_position"]
    ].copy()

    if (
        len(m) != 210
        or m.variant_key.nunique() != 210
        or m.mutation.nunique() != 206
        or m.protein_position.nunique() != 168
    ):
        raise ValueError("Unexpected frozen TRAIN cohort")

    m["mutation"] = m.mutation.astype(str).str.upper()

    for row in m.itertuples():
        match = re.fullmatch(
            r"[A-Z](\d+)[A-Z]", row.mutation
        )
        if not match or int(match.group(1)) != row.protein_position:
            raise ValueError("Mutation-position mismatch")

    matched_keys = set()

    if ETI.exists():
        eti = pd.read_csv(
            ETI,
            usecols=["variant_key", "protein_position"],
            dtype={"variant_key": str}
        )

        if (
            len(eti) != 66
            or eti.variant_key.nunique() != 66
            or eti.protein_position.nunique() != 56
        ):
            raise ValueError("ETI TRAIN cohort changed")

        matched_keys = set(eti.variant_key)

        if not matched_keys.issubset(set(m.variant_key)):
            raise ValueError("Matched ETI ID outside TRAIN")
    else:
        print("WARNING: Phase30C matched TRAIN CSV unavailable")

    print("=" * 95)
    print("PHASE 30I | KROES EXTERNAL ORGANOID FEASIBILITY")
    print("=" * 95)
    print("TRAIN mutation identifiers only.")
    print("Published examples are not the complete cohort.")
    print("VTD organoid outcomes are NOT VTI FRT outcomes.")

    allele_rows = []
    genotype_rows = []

    for genotype, evidence in CASES:
        alleles = genotype.split("/")

        if len(alleles) != 2:
            raise ValueError("Expected two-allele genotype")

        exact_keys = set()
        eti_keys = set()
        missense_count = 0

        for i, raw_allele in enumerate(alleles, 1):
            code, kind = protein_code(raw_allele)

            if kind == "missense":
                missense_count += 1

            matches = (
                m.loc[m.mutation == code]
                if code is not None
                else m.iloc[0:0]
            )

            keys = set(matches.variant_key)
            exact_keys.update(keys)
            eti_keys.update(keys & matched_keys)

            allele_rows.append({
                "genotype": genotype,
                "allele_number": i,
                "published_allele": raw_allele,
                "protein_code": code,
                "allele_type": kind,
                "exact_vti_train_matches": len(keys),
                "matched_eti_train_matches": len(
                    keys & matched_keys
                ),
                "matched_variant_keys": " | ".join(
                    sorted(keys)
                ),
                "published_evidence": evidence,
            })

        genotype_rows.append({
            "published_genotype": genotype,
            "published_evidence": evidence,
            "missense_alleles": missense_count,
            "exact_vti_train_variants": len(exact_keys),
            "exact_eti_train_variants": len(eti_keys),
            "vti_train_variant_keys": " | ".join(
                sorted(exact_keys)
            ),
            "eti_train_variant_keys": " | ".join(
                sorted(eti_keys)
            ),
            "external_numeric_validation_ready": False,
        })

    genotypes = pd.DataFrame(genotype_rows)
    alleles = pd.DataFrame(allele_rows)

    print("\nA. EXACT AMINO-ACID SUBSTITUTION OVERLAP")

    print(
        genotypes[[
            "published_genotype",
            "missense_alleles",
            "exact_vti_train_variants",
            "exact_eti_train_variants",
        ]].to_string(index=False)
    )

    print("\nB. OVERLAP SUMMARY")

    print(
        "Published genotype examples:",
        len(genotypes)
    )

    print(
        "Examples with >=1 exact VTI TRAIN substitution:",
        int((genotypes.exact_vti_train_variants > 0).sum())
    )

    print(
        "Examples with >=1 ETI-matched TRAIN substitution:",
        int((genotypes.exact_eti_train_variants > 0).sum())
    )

    print("\nC. VALIDATION LIMITATIONS")
    print("- Two-allele organoid genotypes are not isolated variants.")
    print("- VTD contains deutivacaftor; Bihler VTI uses ivacaftor.")
    print("- FIS/SLA outcomes differ from FRT %WT function.")
    print("- Per-organoid numerical outcomes not verified.")
    print("- No external MAE or correlation computed.")

    OUT.mkdir(parents=True, exist_ok=True)

    genotypes.to_csv(
        OUT / "phase30i_genotype_overlap.csv",
        index=False
    )

    alleles.to_csv(
        OUT / "phase30i_allele_overlap.csv",
        index=False
    )

    provenance = {
        "publication_doi": "10.1016/j.jcf.2026.06.002",
        "source": "published main-text genotype examples",
        "complete_kroes_cohort": False,
        "quantitative_organoid_data_available": "not verified",
        "independent_prediction_validation": False,
        "protected_labels_accessed": False,
    }

    (
        OUT / "phase30i_provenance.json"
    ).write_text(
        json.dumps(provenance, indent=2) + "\n"
    )

    print("\nPASS Phase30I; outputs:", OUT)


if __name__ == "__main__":
    main()
