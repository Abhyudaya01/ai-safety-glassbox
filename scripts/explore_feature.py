# scripts/explore_feature.py

import os
import sys
import argparse

# --- PATH SETUP ---
CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(CURRENT_DIR, "..", "src")
sys.path.append(os.path.abspath(SRC_DIR))
# ------------------

from glassbox.sae import (
    load_sae,
    default_sae_path,
    get_feature_activations_for_text,
    load_corpus,
)


def main():
    parser = argparse.ArgumentParser(
        description="Show top sentences for a given SAE feature."
    )
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument(
        "--feature",
        type=int,
        required=True,
        help="Feature index (as shown in the dashboard table).",
    )
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional SAE checkpoint path. Defaults to data/cache/sae/sae_<model>_layer<k>.pt",
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        default="data/sae_corpus.txt",
        help="Corpus file (one sentence per line).",
    )

    args = parser.parse_args()

    model_name = args.model
    layer_idx = args.layer
    feature_idx = args.feature
    top_k = args.top_k

    # 1. Load SAE
    if args.checkpoint is None:
        ckpt_path = default_sae_path(model_name, layer_idx)
    else:
        ckpt_path = args.checkpoint

    print(f"Loading SAE from: {ckpt_path}")
    sae = load_sae(model_name, layer_idx, path=ckpt_path)

    # 2. Load corpus
    prompts = load_corpus(args.corpus_path)
    print(f"Loaded {len(prompts)} prompts from {args.corpus_path}")

    # 3. Compute activations for the chosen feature
    results = []
    for text in prompts:
        vec = get_feature_activations_for_text(text, model_name, layer_idx, sae)
        if feature_idx < 0 or feature_idx >= vec.shape[0]:
            raise ValueError(
                f"Feature index {feature_idx} is out of range for this SAE "
                f"(d_hidden = {vec.shape[0]})."
            )
        score = float(vec[feature_idx].item())
        results.append({"text": text, "activation": score})

    # 4. Sort and print top-k
    results.sort(key=lambda x: x["activation"], reverse=True)

    print()
    print(f"=== Top {top_k} sentences for feature {feature_idx} (layer {layer_idx}) ===")
    for i, row in enumerate(results[:top_k], start=1):
        print(f"\n#{i}")
        print(f"Activation: {row['activation']:.4f}")
        print(f"Text      : {row['text']}")


if __name__ == "__main__":
    main()
