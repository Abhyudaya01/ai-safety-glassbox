# scripts/explore_feature.py
import os
import sys
import argparse

# --- PATH SETUP: make sure we can import from src/ ---
CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(CURRENT_DIR, "..", "src")
sys.path.append(os.path.abspath(SRC_DIR))
# -----------------------------------------------------

from glassbox.sae import (
    load_sae,
    default_sae_path,
    get_feature_activations_for_text,
)

# TODO: replace this with your real corpus later (from a file, dataset, etc.)
EXAMPLE_PROMPTS = [
    "I went to Paris and London last summer.",
    "The Eiffel Tower is a famous landmark in France.",
    "Berlin is the capital of Germany.",
    "I wrote some Python code to sort a list.",
    "JavaScript async/await helps with asynchronous programming.",
    "She expressed deep love and care for her family.",
    "New York and Tokyo are large, busy cities.",
    "I visited Rome and saw many historical monuments.",
    "The cat slept peacefully on the sofa.",
    "He refactored his C++ code to improve performance.",
]


def main():
    parser = argparse.ArgumentParser(
        description="Show top sentences for a given SAE feature."
    )
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--feature", type=int, required=True,
                        help="Feature index (as shown in the dashboard table).")
    parser.add_argument("--top_k", type=int, default=10,
                        help="How many top sentences to show.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional SAE checkpoint path. Defaults to data/cache/sae/sae_<model>_layer<k>.pt",
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

    # 2. For each prompt, compute activation of this feature
    results = []
    for text in EXAMPLE_PROMPTS:
        vec = get_feature_activations_for_text(text, model_name, layer_idx, sae)
        if feature_idx < 0 or feature_idx >= vec.shape[0]:
            raise ValueError(
                f"Feature index {feature_idx} is out of range for this SAE "
                f"(d_hidden = {vec.shape[0]})."
            )
        score = float(vec[feature_idx].item())
        results.append({"text": text, "activation": score})

    # 3. Sort by activation, descending
    results.sort(key=lambda x: x["activation"], reverse=True)

    # 4. Print top-k sentences
    print()
    print(f"=== Top {top_k} sentences for feature {feature_idx} (layer {layer_idx}) ===")
    for i, row in enumerate(results[:top_k], start=1):
        print(f"\n#{i}")
        print(f"Activation: {row['activation']:.4f}")
        print(f"Text      : {row['text']}")


if __name__ == "__main__":
    main()
