import os
from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .model_loader import ModelWrapper


# =========================
# Paths & Helpers
# =========================

def default_sae_dir() -> str:
    """Base directory for SAE checkpoints."""
    return os.path.join("data", "cache", "sae")


def default_sae_path(model_name: str, layer_idx: int) -> str:
    """Default path for an SAE trained on (model_name, layer_idx)."""
    fname = f"sae_{model_name}_layer{layer_idx}.pt"
    return os.path.join(default_sae_dir(), fname)


def load_corpus(corpus_path: str) -> List[str]:
    """Load a simple corpus: one sentence per line."""
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    with open(corpus_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    return lines


# =========================
# Sparse Autoencoder
# =========================

class SparseAutoencoder(nn.Module):
    """
    Simple sparse autoencoder:
      encoder: d_model -> d_hidden (features)
      decoder: d_hidden -> d_model (reconstruct activations)

    We use ReLU so most features are 0 (sparse).
    """

    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden

        self.encoder = nn.Linear(d_model, d_hidden)
        self.decoder = nn.Linear(d_hidden, d_model)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [*, d_model]  activations
        returns: [*, d_hidden] feature activations (ReLU)
        """
        return torch.relu(self.encoder(x))

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: [*, d_hidden]
        returns: [*, d_model]
        """
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [*, d_model]
        returns:
            recon: [*, d_model]
            h:     [*, d_hidden]
        """
        h = self.encode(x)
        recon = self.decode(h)
        return recon, h


# =========================
# Training
# =========================

def train_sae_on_layer(
    model_name: str,
    layer_idx: int,
    corpus_path: Optional[str] = None,
    max_tokens: int = 50_000,
    d_hidden: int = 512,
    l1_coef: float = 1e-2,
    epochs: int = 10,
    batch_size: int = 64,
    device: Optional[str] = None,
) -> None:
    """
    Train a sparse autoencoder on residual stream activations at a given layer.

    - model_name: e.g. "gpt2"
    - layer_idx: which transformer block's resid_post to use
    - corpus_path: text file, one sentence per line. If None, uses a tiny toy corpus.
    """

    print(f"[SAE] Training on model={model_name}, layer={layer_idx}")

    # 1. Load model
    print("⬇️ Loading model into RAM...")
    model = ModelWrapper.load(model_name)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"✅ Model {model_name} loaded on {device}.")

    # 2. Load corpus
    if corpus_path is not None:
        print(f"[SAE] Loading corpus from: {corpus_path}")
        prompts = load_corpus(corpus_path)
    else:
        print("[SAE] No corpus_path provided; using small built-in toy corpus.")
        prompts = [
            "I went to Paris and London last summer.",
            "The Eiffel Tower is a famous landmark in France.",
            "Berlin is the capital of Germany.",
            "I wrote some Python code to sort a list.",
            "She expressed deep love and care for her family.",
            "The cat slept peacefully on the sofa.",
        ]

    # 3. Collect activations up to max_tokens
    hook_name = f"blocks.{layer_idx}.hook_resid_post"
    acts_list = []
    token_count = 0

    for text in prompts:
        if token_count >= max_tokens:
            break
        _, cache = model.run_with_cache(text, remove_batch_dim=True)
        acts = cache[hook_name]  # [seq_len, d_model]
        acts = acts.to(device)
        acts_list.append(acts)
        token_count += acts.shape[0]

    if not acts_list:
        raise RuntimeError("No activations collected for SAE training.")

    activations = torch.cat(acts_list, dim=0)  # [total_tokens, d_model]
    d_model = activations.shape[-1]
    print(f"[SAE] Collected activations: {activations.shape}, d_model={d_model}")

    # 4. Build SAE
    sae = SparseAutoencoder(d_model=d_model, d_hidden=d_hidden).to(device)
    optim = torch.optim.Adam(sae.parameters(), lr=1e-3)

    # 5. Train loop
    dataset = TensorDataset(activations)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        sae.train()
        total_loss = 0.0

        for (batch,) in dataloader:
            batch = batch.to(device)
            recon, codes = sae(batch)

            mse = torch.mean((recon - batch) ** 2)
            l1 = torch.mean(torch.abs(codes))
            loss = mse + l1_coef * l1

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item() * batch.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"[SAE layer {layer_idx}] Epoch {epoch}/{epochs} - Loss {avg_loss:.6f}")

    # 6. Save checkpoint
    os.makedirs(default_sae_dir(), exist_ok=True)
    ckpt_path = default_sae_path(model_name, layer_idx)
    torch.save(
        {
            "state_dict": sae.state_dict(),
            "d_model": d_model,
            "d_hidden": d_hidden,
            "layer_idx": layer_idx,
            "model_name": model_name,
        },
        ckpt_path,
    )
    print(f"Saved SAE to: {ckpt_path}")


# =========================
# Loading & Inference
# =========================

def _sae_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_sae(model_name: str, layer_idx: int, path: Optional[str] = None) -> SparseAutoencoder:
    """
    Load a trained SAE checkpoint for (model_name, layer_idx).
    """
    if path is None:
        path = default_sae_path(model_name, layer_idx)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find SAE checkpoint at {path}. "
            f"Make sure you trained it (e.g. python -m scripts.train_sae)."
        )

    ckpt = torch.load(path, map_location=_sae_device())
    d_model = ckpt["d_model"]
    d_hidden = ckpt["d_hidden"]

    sae = SparseAutoencoder(d_model=d_model, d_hidden=d_hidden)
    sae.load_state_dict(ckpt["state_dict"])
    sae.to(_sae_device())
    sae.eval()
    return sae


@torch.no_grad()
def _encode_text_to_features(
    text: str,
    model_name: str,
    layer_idx: int,
    sae: SparseAutoencoder,
) -> torch.Tensor:
    """
    Helper: run text through model, get layer activations,
    encode with SAE, max-pool over tokens.

    Returns: [d_hidden] pooled feature activations (on CPU).
    """
    model = ModelWrapper.load(model_name)
    model.to(_sae_device())

    _, cache = model.run_with_cache(text, remove_batch_dim=True)
    hook_name = f"blocks.{layer_idx}.hook_resid_post"
    acts = cache[hook_name]  # [seq_len, d_model]
    acts = acts.to(_sae_device())

    codes = sae.encode(acts)  # [seq_len, d_hidden]
    pooled, _ = codes.max(dim=0)  # [d_hidden]
    return pooled.cpu()


@torch.no_grad()
def get_feature_activations_for_text(
    text: str,
    model_name: str,
    layer_idx: int,
    sae: SparseAutoencoder,
) -> torch.Tensor:
    """
    Public wrapper: returns [d_hidden] feature activations for a given text.
    """
    return _encode_text_to_features(text, model_name, layer_idx, sae)


@torch.no_grad()
def get_top_features_for_text(
    text: str,
    model_name: str,
    layer_idx: int,
    sae: SparseAutoencoder,
    top_k: int = 20,
) -> List[dict]:
    """
    Return top-k most active features for a given text as a list of dicts:
        [{"feature": idx, "activation": value}, ...]
    """
    pooled = _encode_text_to_features(text, model_name, layer_idx, sae)
    d_hidden = pooled.shape[0]
    k = min(top_k, d_hidden)

    values, indices = torch.topk(pooled, k=k)
    results = []
    for i in range(k):
        results.append(
            {
                "feature": int(indices[i].item()),
                "activation": float(values[i].item()),
            }
        )
    return results
