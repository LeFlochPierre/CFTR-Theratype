#!/usr/bin/env python3
"""Phase 27D-4A: label-free, mutation-centred frozen ESM-2 features.

Only uses mutation identifiers from VTI TRAIN and the canonical CFTR reference.
Does not access VAL/TEST labels, train predictors, or fit PCA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results/biology/phase27d"
FASTA = ROOT / "data/external/cftr_structures/P13569.fasta"
MUT_RE = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])$")


def read_fasta(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing reference FASTA: {path}")
    s = "".join(line.strip() for line in path.read_text().splitlines() if not line.startswith(">"))
    if len(s) != 1480 or not s.isalpha():
        raise ValueError(f"Unexpected CFTR reference length/content: {len(s)}")
    return s.upper()


def local_pair(ref: str, mutation: str, window_size: int = 511):
    match = MUT_RE.fullmatch(mutation)
    if not match:
        raise ValueError(f"Not a canonical missense substitution: {mutation}")
    old, number, new = match.groups()
    p = int(number)
    if not 1 <= p <= len(ref) or ref[p - 1] != old or old == new:
        raise ValueError(f"Reference mismatch or non-missense: {mutation}")
    if window_size < 5 or window_size > 1022:
        raise ValueError("Window must contain 5 to 1022 residues")
    start = max(0, min(p - 1 - window_size // 2, len(ref) - window_size))
    end = min(len(ref), start + window_size)
    wt = ref[start:end]
    offset = (p - 1) - start
    mutant = wt[:offset] + new + wt[offset + 1:]
    assert wt[offset] == old and mutant[offset] == new and len(mutant) == len(wt)
    return wt, mutant, offset, start + 1, end


def get_train_mutations():
    import phase26h_auxiliary_corrector as h
    from phase26k_trafficking_preflight import mutation_code
    v, _, _, _, _ = h.load_data()
    if len(v) != 210 or v.variant_key.astype(str).nunique() != 210:
        raise ValueError("Unexpected VTI training cohort")
    candidates = [c for c in ("hgvs_protein", "mutation", "variant_key", "protein_change", "variant") if c in v]
    if not candidates:
        raise ValueError(f"No mutation columns in VTI training metadata: {list(v.columns)}")
    parsed = []
    for _, row in v.iterrows():
        values = [row[c] for c in candidates if pd.notna(row[c])]
        codes = [mutation_code(str(q)) for q in values]
        codes = [q for q in codes if isinstance(q, str) and MUT_RE.fullmatch(q)]
        if len(set(codes)) > 1:
            raise ValueError(f"Conflicting parsed mutations: {codes}")
        if not codes:
            raise ValueError(f"Unparseable substitution from VTI TRAIN: {values}")
        parsed.append(codes[0])
    v = v.copy()
    v["mutation"] = parsed
    position = v.mutation.str.extract(r"[A-Z]([0-9]+)[A-Z]", expand=False).astype(int)
    if not np.array_equal(position.to_numpy(), v.protein_position.to_numpy(int)):
        raise ValueError("Mutation position mismatch")
    return v[["variant_key", "protein_position", "mutation"]].copy()


def compute_one(model, tokenizer, device, ref, mutation, window_size):
    import torch
    wt, mt, j, start, end = local_pair(ref, mutation, window_size)
    enc = tokenizer([wt, mt], return_tensors="pt", padding=True, add_special_tokens=True)
    token_i = j + 1  # ESM tokenizer inserts <cls> before residues.
    for ix, aa in ((0, wt[j]), (1, mt[j])):
        token_id = int(enc["input_ids"][ix, token_i])
        if token_id != tokenizer.convert_tokens_to_ids(aa):
            raise ValueError(f"Token-to-residue alignment failed at {mutation}")
    enc = {k: t.to(device) for k, t in enc.items()}
    with torch.inference_mode():
        outputs = model(**enc, output_hidden_states=True)
        hidden = outputs.hidden_states[-1].float().cpu().numpy()
        h_wt = hidden[0, token_i, :]
        h_mut = hidden[1, token_i, :]
        delta = (h_mut - h_wt).astype(np.float32)
        denom = np.linalg.norm(h_wt) * np.linalg.norm(h_mut)
        cosine = float(np.dot(h_wt, h_mut) / denom) if denom > 0 else np.nan
        masked = {k: t[:1].clone() for k, t in enc.items()}
        masked["input_ids"][0, token_i] = tokenizer.mask_token_id
        logits = model(**masked).logits[0, token_i].float()
        logp = torch.log_softmax(logits, dim=-1)
        old, _, new = MUT_RE.fullmatch(mutation).groups()
        llr = float((logp[tokenizer.convert_tokens_to_ids(new)] -
                     logp[tokenizer.convert_tokens_to_ids(old)]).cpu())
        mask_entropy = float((-(logp.exp() * logp).sum()).cpu())
    return {
        "mutation": mutation, "protein_position": int(MUT_RE.fullmatch(mutation).group(2)),
        "window_start": start, "window_end": end, "window_offset": j,
        "esm2_llr": llr, "esm2_mask_entropy": mask_entropy,
        "esm2_delta_l2": float(np.linalg.norm(delta)),
        "esm2_site_cosine": cosine,
    }, delta


def self_test():
    ref = "ACDEFGHIKLMNPQRSTVWY" * 74
    assert len(ref) == 1480
    for mutation in ("A1C", "Y20A", "A1001C", "Y1480A"):
        a, b, j, start, end = local_pair(ref, mutation, 511)
        assert len(a) == len(b) == 511
        assert end - start + 1 == 511
        assert start <= int(MUT_RE.fullmatch(mutation).group(2)) <= end
        assert sum(x != y for x, y in zip(a, b)) == 1
    print("PASS synthetic: canonical mutation, reference alignment, centred 511-AA windows")


def main(checkpoint: str, window: int):
    self_test()
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    ref = read_fasta(FASTA)
    meta = get_train_mutations()
    mutation_codes = sorted(meta.mutation.unique())
    print("=" * 90)
    print("PHASE 27D-4A | FROZEN ESM-2 MUTATION FEATURES | LABEL-FREE")
    print("=" * 90)
    print(f"VTI TRAIN rows={len(meta)}, unique AA substitutions={len(mutation_codes)}")
    print(f"Protein reference length={len(ref)}; window={window} residues")
    print(f"Pretrained checkpoint: {checkpoint}")
    print("No supervised fitting; no VAL/TEST labels accessed")
    for mutation in mutation_codes:
        local_pair(ref, mutation, window)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}; loading frozen ESM-2 ...", flush=True)
    tok = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForMaskedLM.from_pretrained(checkpoint).to(device).eval()
    rows, deltas = [], []
    for i, mut in enumerate(mutation_codes, 1):
        row, d = compute_one(model, tok, device, ref, mut, window)
        rows.append(row)
        deltas.append(d)
        if i == 1 or i % 25 == 0 or i == len(mutation_codes):
            print(f"processed {i}/{len(mutation_codes)} substitutions", flush=True)
    df = pd.DataFrame(rows)
    D = np.stack(deltas)
    if not np.isfinite(D).all() or not np.isfinite(df[["esm2_llr", "esm2_delta_l2", "esm2_site_cosine"]]).all().all():
        raise ValueError("Nonfinite ESM features")
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "phase27d4a_esm2_scalar_features.csv", index=False)
    np.savez_compressed(OUT / "phase27d4a_esm2_delta_embeddings.npz",
                        mutation=np.asarray(mutation_codes), delta=D)
    manifest = {
        "model": checkpoint, "window": window, "reference_sha256": hashlib.sha256(ref.encode()).hexdigest(),
        "n_rows": len(meta), "n_mutations": len(mutation_codes), "embedding_shape": list(D.shape),
        "data_scope": "VTI TRAIN mutation identities only, no outcomes",
    }
    (OUT / "phase27d4a_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("\nA. FEATURE QC")
    print(f"Delta embeddings: shape={D.shape}, std={D.std():.5f}, finite={np.isfinite(D).all()}")
    print(df[["esm2_llr", "esm2_delta_l2", "esm2_site_cosine"]].describe().round(4).to_string())
    print("\nB. COVERAGE QC")
    print(f"Protein positions > 1022: {int((df.protein_position > 1022).sum())} included via local windows")
    print(f"Unique mutations: {len(df)}, mapped TRAIN records: {len(meta)}")
    print("\nPASS: frozen ESM-2 features saved; no predictive model trained.")
    print("Files:", OUT / "phase27d4a_esm2_scalar_features.csv", OUT / "phase27d4a_esm2_delta_embeddings.npz")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--model", default="facebook/esm2_t12_35M_UR50D")
    p.add_argument("--window", type=int, default=511)
    args = p.parse_args()
    if args.self_test: self_test()
    else: main(args.model, args.window)
