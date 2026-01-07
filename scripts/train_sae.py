# scripts/train_sae.py
import os
import sys
import argparse


CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(CURRENT_DIR, "..", "src")
sys.path.append(os.path.abspath(SRC_DIR))

from glassbox.sae import train_sae_on_layer

EXAMPLE_PROMPTS = [
    "I went to Paris and London last summer.",
    "The Eiffel Tower is a famous landmark in France.",
    "for (int i = 0; i < n; i++) { sum += i; }",
    "In Python, you can define a function using the def keyword.",
    "She expressed deep love and compassion for her friends.",
    "The capital of Germany is Berlin.",
    "JavaScript async await makes asynchronous code easier to read.",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--l1", type=float, default=1e-2)
    parser.add_argument("--max_tokens", type=int, default=50_000)
    args = parser.parse_args()

    train_sae_on_layer(
        prompts=EXAMPLE_PROMPTS,
        model_name=args.model,
        layer_idx=args.layer,
        d_hidden=args.hidden,
        num_epochs=args.epochs,
        l1_coef=args.l1,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
