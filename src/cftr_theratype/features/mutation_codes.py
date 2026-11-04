"""Conservative CFTR canonical missense extraction used in Phase26K/27D.

Unlike exact-ID normalization, this parser can find an embedded substitution
inside a longer free-text mutation annotation; do not merge these semantics.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd


AA = dict(zip(
    [
        "Ala", "Arg", "Asn", "Asp", "Cys",
        "Glu", "Gln", "Gly", "His", "Ile",
        "Leu", "Lys", "Met", "Phe", "Pro",
        "Ser", "Thr", "Trp", "Tyr", "Val"
    ],
    list("ARNDCEQGHILKMFPSTWYV")
))

ALLOWED = set(AA.values())

PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:p\.)?"
    r"([A-Za-z]{3}|[A-Za-z])"
    r"([1-9]\d{0,3})"
    r"([A-Za-z]{3}|[A-Za-z])"
    r"(?=$|[^A-Za-z0-9])",
    re.I
)


def aa_symbol(token):
    if len(token) == 1:
        return (
            token.upper()
            if token.upper() in ALLOWED
            else None
        )
    return AA.get(token.capitalize())


def mutation_code(value):
    if pd.isna(value):
        return None

    for m in PATTERN.finditer(str(value).strip()):
        a = aa_symbol(m.group(1))
        b = aa_symbol(m.group(3))
        pos = int(m.group(2))

        if a and b and a != b and 1 <= pos <= 1480:
            return f"{a}{pos}{b}"

    return None


def position_from_code(code):
    if not isinstance(code, str):
        return np.nan

    m = re.fullmatch(
        r"[A-Z]([0-9]{1,4})[A-Z]", code
    )
    return int(m.group(1)) if m else np.nan


def extract_codes(frame, pp):
    candidates = [c for c in ('mutation', 'hgvs_protein', 'protein_change',
                              'variant_key') if c in frame.columns]
    for column in candidates:
        codes = [mutation_code(str(x)) for x in frame[column].astype(str)]
        good = [bool(re.fullmatch(r'[A-Z]\d+[A-Z]', str(x))) for x in codes]
        if all(good):
            parsed_pp = [int(re.search(r'\d+', x).group()) for x in codes]
            if np.array_equal(parsed_pp, pp):
                print('Mutation-code source:', column)
                return codes
    raise ValueError('Could not parse all missense protein substitutions. '
                     f'Available columns: {list(frame.columns)}. '
                     'Inspect the original VTI TRAIN fields; do not guess amino acids.')
