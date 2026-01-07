# src/glassbox/sae.py
import os
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .model_loader import ModelWrapper


class SparseAutoencoder(nn.Module):
    """
    Simple Sparse Autoencoder:
      encoder: d_model -> d_hidden
      decoder: d_hidden -> d_model
      codes:   ReLU + L1 sparsity during training
    """
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.encoder = nn.Linear(d_model, d_hidden, bias=True)
        self.decoder = nn.Linear(d_hidden, d_model, bias=True)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, d_model]
        return torch.relu(self.encoder(x))

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encode(x)
        recon = self.decode(h)
        return recon, h


def _collect_activations(
    prompts: List[str],
    model_name: str,
    layer_idx: int,
    max_tokens: int = 50_000,
) -> torch.Tensor:
    """
    Run prompts, collect residual stream at one layer, and subsample tokens.
    Returns [num_tokens, d_model] on CPU.
    """
    model = ModelWrapper.load(model_name)
    hook_name = f"blocks.{layer_idx}.hook_resid_post"

    all_acts = []
    for text in prompts:
        _, cache = model.run_with_cache(text, remove_batch_dim=True)
        layer_act = cache[hook_name]          # [seq_len, d_model]
        all_acts.append(layer_act.detach())

    acts = torch.cat(all_acts, dim=0)         # [total_tokens, d_model]

    if acts.size(0) > max_tokens:
        idx = torch.randperm(acts.size(0))[:max_tokens]
        acts = acts[idx]

    return acts.cpu()


def train_sae_on_layer(
    prompts: List[str],
    model_name: str,
    layer_idx: int,
    d_hidden: int = 4096,
    batch_size: int = 256,
    num_epochs: int = 5,
    lr: float = 1e-3,
    l1_coef: float = 1e-3,
    max_tokens: int = 50_000,
    save_dir: str = "data/cache/sae",
) -> SparseAutoencoder:
    """
    Train a Sparse Autoencoder on residual stream of one transformer layer.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    acts = _collect_activations(prompts, model_name, layer_idx, max_tokens=max_tokens)
    d_model = acts.size(1)

    dataset = TensorDataset(acts)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    sae = SparseAutoencoder(d_model=d_model, d_hidden=d_hidden).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)

    for epoch in range(num_epochs):
        sae.train()
        total_loss = 0.0

        for (batch,) in loader:
            batch = batch.to(device)
            recon, codes = sae(batch)

            mse = torch.mean((recon - batch) ** 2)
            l1 = torch.mean(torch.abs(codes))

            loss = mse + l1_coef * l1
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * batch.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"[SAE layer {layer_idx}] Epoch {epoch+1}/{num_epochs} - Loss {avg_loss:.6f}")

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        save_dir, f"sae_{model_name.replace('/', '_')}_layer{layer_idx}.pt"
    )

    torch.save(
        {
            "state_dict": sae.state_dict(),
            "d_model": d_model,
            "d_hidden": sae.encoder.out_features,
            "model_name": model_name,
            "layer_idx": layer_idx,
        },
        save_path,
    )
    print(f"Saved SAE to: {save_path}")
    return sae


def default_sae_path(model_name: str, layer_idx: int, save_dir: str = "data/cache/sae"):
    return os.path.join(save_dir, f"sae_{model_name.replace('/', '_')}_layer{layer_idx}.pt")


def load_sae(model_name: str, layer_idx: int, path: str = None) -> SparseAutoencoder:
    """
    Load a trained SAE checkpoint for given model + layer.
    """
    if path is None:
        path = default_sae_path(model_name, layer_idx)

    ckpt = torch.load(path, map_location="cpu")
    d_model = ckpt["d_model"]
    d_hidden = ckpt["d_hidden"]

    sae = SparseAutoencoder(d_model=d_model, d_hidden=d_hidden)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()
    return sae


def get_feature_activations_for_text(
    text: str,
    model_name: str,
    layer_idx: int,
    sae: SparseAutoencoder,
) -> torch.Tensor:
    """
    For a given text, return a [d_hidden] vector of SAE feature activations
    (max-pooled over the sequence).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelWrapper.load(model_name)

    _, cache = model.run_with_cache(text, remove_batch_dim=True)
    hook_name = f"blocks.{layer_idx}.hook_resid_post"
    acts = cache[hook_name]                   # [seq_len, d_model]

    sae = sae.to(device)
    with torch.no_grad():
        codes = sae.encode(acts.to(device))   # [seq_len, d_hidden]
        pooled, _ = codes.max(dim=0)          # [d_hidden]

    return pooled.cpu()


def get_top_features_for_text(
    text: str,
    model_name: str,
    layer_idx: int,
    sae: SparseAutoencoder,
    k: int = 20,
):
    """
    Return list of {feature, activation} for top-k SAE features for this text.
    """
    vec = get_feature_activations_for_text(text, model_name, layer_idx, sae)
    values, indices = torch.topk(vec, k=k)

    return [
        {"feature": int(idx.item()), "activation": float(val.item())}
        for idx, val in zip(indices, values)
    ]
