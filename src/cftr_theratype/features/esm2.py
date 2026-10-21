"""Frozen ESM-2 embeddings for CFTR missense variants.

No labels, optimisation or model fine-tuning are involved.
Only the original training and validation partitions are processed.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cftr_theratype.reference import bihler_2024_assay_sequence


ROOT = Path(__file__).resolve().parents[3]
SPLITS = ROOT / "data/processed/bihler_2024_missense_splits.parquet"
PAIRS = ROOT / "data/processed/bihler_2024_theratyping.parquet"
PROTOCOL = ROOT / "data/manifests/split_protocol.json"
OUTPUT = ROOT / "data/interim/esm2_35m_development.npz"

MODEL_ID = "facebook/esm2_t12_35M_UR50D"
WINDOW = 801


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mutation_windows(sequence, position, alternate, window=WINDOW):
    """Create aligned WT/mutant windows with the same residue index."""

    position = int(position)
    index = position - 1

    if not 0 <= index < len(sequence):
        raise ValueError(f"Invalid CFTR position: {position}")

    if alternate == sequence[index]:
        raise ValueError("Mutation is identical to reference residue.")

    if not 1 <= window <= len(sequence):
        raise ValueError("Invalid window length.")

    start = max(
        0,
        min(index - window // 2, len(sequence) - window),
    )

    wt = sequence[start:start + window]
    local_index = index - start

    mutant = (
        wt[:local_index]
        + alternate
        + wt[local_index + 1:]
    )

    assert len(wt) == len(mutant) == window
    assert sum(a != b for a, b in zip(wt, mutant)) == 1

    return wt, mutant, local_index


def extract_embeddings(smoke=False):
    import torch
    from huggingface_hub import model_info
    from transformers import AutoModel, AutoTokenizer

    print("=" * 85)
    print("CFTR-THERATYPE | FROZEN ESM-2 MUTATION EMBEDDINGS")
    print("=" * 85)

    protocol = json.loads(PROTOCOL.read_text())

    if sha256(PAIRS) != protocol["source_sha256"]:
        raise ValueError("Source dataset differs from frozen split protocol.")

    splits = pd.read_parquet(SPLITS)

    if splits["variant_key"].duplicated().any():
        raise ValueError("Duplicate variant keys.")

    if set(splits["split"]) != {"train", "val", "test"}:
        raise ValueError("Expected train/val/test partitions.")

    development = (
        splits.loc[splits["split"].isin(["train", "val"])]
        .sort_values(["split", "variant_key"])
        .reset_index(drop=True)
    )

    if smoke:
        development = development.head(3)

    sequence = bihler_2024_assay_sequence()
    keys = development["variant_key"].astype(str).tolist()

    if development.empty:
        raise ValueError("No eligible development variants.")

    print("\n[1] EMBEDDING CONFIGURATION")
    print(f"Model:              {MODEL_ID}")
    print(f"Mutation window:    {WINDOW} amino acids")
    print(f"Variants:           {len(development)}")
    print(f"Mode:               {'SMOKE TEST' if smoke else 'FULL EXTRACTION'}")
    print("Test partition:     excluded")
    print("Supervised labels:  not loaded")

    # Resolve the model repository revision for reproducibility.
    revision = model_info(MODEL_ID).sha

    metadata = {
        "model_id": MODEL_ID,
        "model_revision": revision,
        "window": WINDOW,
        "sequence_sha256": hashlib.sha256(
            sequence.encode()
        ).hexdigest(),
        "split_sha256": sha256(SPLITS),
        "dataset_sha256": sha256(PAIRS),
        "num_variants": len(keys),
        "mode": "development_only",
    }

    # Reuse a compatible embedding cache rather than re-running ESM-2.
    if OUTPUT.exists() and not smoke:
        with np.load(OUTPUT, allow_pickle=False) as cached:
            previous = json.loads(str(cached["metadata"]))
            previous_keys = cached["keys"].tolist()

            if previous == metadata and previous_keys == keys:
                print("\nPASS: Existing embedding cache is valid.")
                print(f"Cached shape: {cached['delta'].shape}")
                return

        raise ValueError(
            "Existing embedding cache uses different inputs or model. "
            "Review provenance before regenerating it."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\n[2] LOADING MODEL")
    print(f"Resolved revision: {revision}")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=revision
    )

    model = AutoModel.from_pretrained(
        MODEL_ID, revision=revision
    ).to(device).eval()

    if WINDOW + 2 > model.config.max_position_embeddings:
        raise ValueError(
            "Window exceeds model positional capacity."
        )

    def encode_residue(protein, local_index):
        """Return final-layer embedding at the mutated residue."""

        encoded = tokenizer(
            protein,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=False,
        )

        # ESM uses one BOS token followed by one token per amino acid.
        token_index = local_index + 1

        if encoded["input_ids"].shape[1] != len(protein) + 2:
            raise ValueError("Unexpected ESM tokenisation.")

        token_id = int(encoded["input_ids"][0, token_index])
        token = tokenizer.convert_ids_to_tokens(token_id)

        if token != protein[local_index]:
            raise ValueError(
                f"Token alignment failed: {token} != "
                f"{protein[local_index]}"
            )

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        with torch.inference_mode():
            hidden = model(**encoded).last_hidden_state
            vector = hidden[0, token_index].float().cpu().numpy()

        if not np.isfinite(vector).all():
            raise ValueError("Non-finite ESM embedding.")

        return vector.astype(np.float32)

    wt_vectors = []
    mutant_vectors = []

    # Reuse WT inference for multiple substitutions at the same position.
    wt_cache = {}

    print("\n[3] EXTRACTING CONTEXTUAL REPRESENTATIONS")

    for i, row in enumerate(development.itertuples(index=False), 1):
        position = int(row.protein_position)
        wt_aa = str(row.reference_aa)
        alternate = str(row.alternate_aa)

        if sequence[position - 1] != wt_aa:
            raise ValueError(
                f"Reference mismatch: {row.variant_key}"
            )

        wt, mutant, local_index = mutation_windows(
            sequence, position, alternate
        )

        if position not in wt_cache:
            wt_cache[position] = encode_residue(
                wt, local_index
            )

        wt_vector = wt_cache[position]
        mutant_vector = encode_residue(
            mutant, local_index
        )

        wt_vectors.append(wt_vector)
        mutant_vectors.append(mutant_vector)

        if i % 25 == 0 or i == len(development):
            print(
                f"Processed {i}/{len(development)} variants | "
                f"WT contexts cached: {len(wt_cache)}",
                flush=True,
            )

    wt_vectors = np.stack(wt_vectors)
    mutant_vectors = np.stack(mutant_vectors)
    delta = mutant_vectors - wt_vectors

    if not (
        np.isfinite(wt_vectors).all()
        and np.isfinite(mutant_vectors).all()
        and np.isfinite(delta).all()
    ):
        raise ValueError("Embedding integrity check failed.")

    print("\n[4] EMBEDDING SUMMARY")
    print(f"WT embeddings:       {wt_vectors.shape}")
    print(f"Mutant embeddings:   {mutant_vectors.shape}")
    print(f"Delta embeddings:    {delta.shape}")
    print(f"Mean delta L2 norm:  {np.linalg.norm(delta, axis=1).mean():.4f}")

    if not smoke:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            OUTPUT,
            keys=np.asarray(keys),
            wt=wt_vectors,
            mutant=mutant_vectors,
            delta=delta,
            metadata=json.dumps(metadata),
        )

        print(f"\nSaved: {OUTPUT.relative_to(ROOT)}")
    else:
        print("\nSmoke test: no embedding cache written.")

    print("ESM EMBEDDING EXTRACTION COMPLETE")
    print("=" * 85)
