import sys
import os
import torch
import argparse
from datasets import load_dataset

# Path setup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glassbox.model_loader import ModelWrapper
from glassbox.sae import SparseAutoencoder, get_activations, train_step


def train_sae(model_name="gpt2", layer_idx=6, n_steps=5000):
    """
    Trains an SAE for a specific model and layer using WikiText data.
    """
    # 1. Load the specific model
    print(f"🧠 Loading {model_name}...")
    model = ModelWrapper.load(model_name)

    # --- FIX: DETECT DEVICE (MPS/CUDA/CPU) ---
    # We need to ensure the SAE lives on the same chip as the main model
    device = model.cfg.device
    print(f"⚙️  Model is running on device: {device}")
    # -----------------------------------------

    # 2. Setup SAE
    d_model = model.cfg.d_model
    sae = SparseAutoencoder(input_dim=d_model, hidden_dim=d_model * 8)

    # --- FIX: MOVE SAE TO GPU ---
    sae = sae.to(device)
    # ----------------------------

    optimizer = torch.optim.Adam(sae.parameters(), lr=3e-4)

    # --- REAL DATA LOADING ---
    print("📚 Loading WikiText-2 dataset (Real English Text)...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    real_text_data = [row['text'] for row in dataset if len(row['text']) > 50]
    print(f"✅ Loaded {len(real_text_data)} real paragraphs for training.")
    # -------------------------

    # 3. Training Loop
    print(f"🚀 Training on {model_name} Layer {layer_idx} (d_model={d_model})...")

    for i in range(n_steps):
        txt = real_text_data[i % len(real_text_data)]
        if len(txt) > 1000:
            txt = txt[:1000]

        # acts will be on the correct device (MPS/GPU) automatically
        acts = get_activations(txt, model, layer_idx)

        loss = train_step(sae, acts, optimizer)

        if i % 100 == 0:
            print(f"Step {i}: Loss {loss.item():.4f}")

    # 4. Save
    os.makedirs("data", exist_ok=True)
    save_path = f"data/sae_{model_name}_layer{layer_idx}.pt"
    torch.save(sae.state_dict(), save_path)
    print(f"✅ Saved SAE to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2", help="gpt2, gpt2-medium, or gpt2-large")
    parser.add_argument("--layer", type=int, default=6, help="Layer index")
    parser.add_argument("--steps", type=int, default=2000, help="Training steps")

    args = parser.parse_args()

    train_sae(model_name=args.model, layer_idx=args.layer, n_steps=args.steps)