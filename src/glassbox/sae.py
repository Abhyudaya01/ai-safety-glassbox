import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # 1. Encode: Linear -> ReLU (to enforce sparsity)
        encoded = F.relu(self.encoder(x))
        # 2. Decode: Linear back to original shape
        decoded = self.decoder(encoded)
        return encoded, decoded


def get_activations(text, model, layer_idx):
    """
    Runs the model and gets the residual stream activations at the specified layer
    using HookedTransformer's native caching mechanism.
    """
    # 1. Run with cache (Native method)
    # This handles tokenization, GPU moving, and extraction automatically.
    # It returns (logits, cache_dictionary)
    _, cache = model.run_with_cache(text)

    # 2. Extract the specific layer's residual stream
    # The standard name for the output of a layer is: 'blocks.{layer}.hook_resid_post'
    hook_name = f"blocks.{layer_idx}.hook_resid_post"

    # Safety check
    if hook_name not in cache:
        raise ValueError(f"Could not find hook {hook_name} in model cache.")

    activations = cache[hook_name]

    # 3. Remove batch dimension: [1, seq_len, d_model] -> [seq_len, d_model]
    return activations.squeeze(0)


def train_step(sae, activations, optimizer, l1_coeff=3e-4):
    """
    Performs one step of SAE training.
    """
    optimizer.zero_grad()

    # Forward pass
    encoded, decoded = sae(activations)

    # 1. Reconstruction Loss (MSE) - Did we keep the info?
    reconstruction_loss = F.mse_loss(decoded, activations)

    # 2. Sparsity Loss (L1) - Did we use few neurons?
    l1_loss = torch.norm(encoded, 1)

    # Combine
    loss = reconstruction_loss + (l1_coeff * l1_loss)

    loss.backward()
    optimizer.step()

    return loss


# --- INFERENCE HELPERS ---

def load_sae(model_name, layer_idx, path=None):
    """
    Loads a trained SAE from disk and moves it to the correct device (MPS/GPU).
    """
    import os
    # Default path if none provided
    if path is None:
        path = f"data/sae_{model_name}_layer{layer_idx}.pt"

    if not os.path.exists(path):
        return None

    # Determine dimensions based on model name
    if "medium" in model_name:
        d_model = 1024
    elif "large" in model_name:
        d_model = 1280
    elif "xl" in model_name:
        d_model = 1600
    else:
        d_model = 768  # Small (default)

    hidden_dim = d_model * 8

    sae = SparseAutoencoder(d_model, hidden_dim)

    # 1. Load weights (defaulting to CPU first for safety)
    sae.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    # 2. AUTO-MOVE TO DEVICE (The Fix!) 🚀
    if torch.backends.mps.is_available():
        sae = sae.to("mps")
    elif torch.cuda.is_available():
        sae = sae.to("cuda")

    return sae


def get_sae_feature_vector(feature_idx, model_name, layer_idx, sae_path=None):
    """
    Extracts the decoder direction (vector) for a specific feature index.
    This vector can be used for steering.
    """
    sae = load_sae(model_name, layer_idx, path=sae_path)
    if sae is None:
        return None

    with torch.no_grad():
        # The decoder weight matrix is [input_dim, hidden_dim]
        # We want the column corresponding to feature_idx
        vector = sae.decoder.weight[:, feature_idx]

    return vector


def get_top_features_for_text(text, model_name, layer_idx, sae, top_k=10):
    """
    Runs text through the SAE and finds which features fired the strongest.
    """
    # 1. Get raw model activations
    # We need a temporary model wrapper or assume one is passed?
    # For simplicity, we load a fresh one here or rely on the caller.
    # ideally, pass 'model' object to avoid reloading.
    # BUT, to keep this function self-contained for the UI:
    from glassbox.model_loader import ModelWrapper
    model = ModelWrapper.load(model_name)

    acts = get_activations(text, model, layer_idx)

    # 2. Run SAE
    with torch.no_grad():
        encoded, _ = sae(acts)
        # Max over the sequence length (find max activation of feature in the sentence)
        # encoded shape: [seq_len, hidden_dim]
        max_activations, _ = torch.max(encoded, dim=0)  # [hidden_dim]

    # 3. Find Top K
    values, indices = torch.topk(max_activations, top_k)

    results = []
    for val, idx in zip(values, indices):
        if val.item() > 0.001:  # Filter dead features
            results.append({"feature_id": idx.item(), "activation": val.item()})

    return results


def get_feature_activations_for_text(text, model_name, layer_idx, sae):
    """
    Returns the max activation vector for a text (helper for corpus scanning).
    """
    from glassbox.model_loader import ModelWrapper
    model = ModelWrapper.load(model_name)
    acts = get_activations(text, model, layer_idx)
    with torch.no_grad():
        encoded, _ = sae(acts)
        max_vals, _ = torch.max(encoded, dim=0)
    return max_vals
