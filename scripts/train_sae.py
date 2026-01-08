# scripts/train_sae.py

import os
import sys
import argparse

# --- PATH SETUP ---
CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(CURRENT_DIR, "..", "src")
sys.path.append(os.path.abspath(SRC_DIR))
# ------------------

from glassbox.sae import train_sae_on_layer


def main():
    parser = argparse.ArgumentParser(
        description="Train a Sparse Autoencoder on GPT-2 layer activations."
    )
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument(
        "--corpus_path",
        type=str,
        default="data/sae_corpus.txt",
        help="Text file with one sentence per line for SAE training.",
    )
    parser.add_argument("--hidden", type=int, default=512, help="SAE hidden size (number of features).")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--l1",
        type=float,
        default=1e-2,
        help="L1 coefficient for sparsity (higher -> more sparse).",
    )
    parser.add_argument("--max_tokens", type=int, default=50_000)
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()

    train_sae_on_layer(
        model_name=args.model,
        layer_idx=args.layer,
        corpus_path=args.corpus_path,
        max_tokens=args.max_tokens,
        d_hidden=args.hidden,
        l1_coef=args.l1,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
