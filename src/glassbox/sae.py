import torch
import os
from transformer_lens import utils

# Define the SAE class (AutoEncoder)
class AutoEncoder(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.d_hidden = cfg["d_mlp"]
        self.d_input = cfg["d_model"]
        self.l1_coeff = cfg["l1_coeff"]
        self.W_enc = torch.nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(self.d_input, self.d_hidden)))
        self.b_enc = torch.nn.Parameter(torch.zeros(self.d_hidden))
        self.W_dec = torch.nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(self.d_hidden, self.d_input)))
        self.b_dec = torch.nn.Parameter(torch.zeros(self.d_input))
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

    def forward(self, x):
        x_cent = x - self.b_dec
        acts = torch.relu(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0)
        l1_loss = self.l1_coeff * (acts.float().abs().sum())
        loss = l2_loss + l1_loss
        return loss, x_reconstruct, acts, l2_loss, l1_loss

    def encode(self, x):
        # Helper to just get activations
        x_cent = x - self.b_dec
        acts = torch.relu(x_cent @ self.W_enc + self.b_enc)
        return acts

def load_sae(model_name="gpt2-small", layer=6, path=None):
    """
    Safely attempts to load the SAE model weights.
    Returns None if file is missing.
    """
    # 1. Construct default path if not provided
    if path is None:
        path = f"src/glassbox/sae_{model_name.replace('-','_')}_layer{layer}.pt"
    
    # 2. Check if file exists
    if not os.path.exists(path):
        # print(f"Warning: SAE weights not found at {path}")
        return None

    # 3. Load
    try:
        # We need the config to initialize the class
        # Ideally, you save the config with the weights or hardcode defaults for now
        cfg = {"d_mlp": 24576, "d_model": 768, "l1_coeff": 3e-4} # Default for GPT2-Small
        sae = AutoEncoder(cfg)
        
        state_dict = torch.load(path, map_location="cpu")
        sae.load_state_dict(state_dict)
        return sae
    except Exception as e:
        print(f"Error loading SAE: {e}")
        return None

def get_top_features_for_text(text, model, layer, sae, top_k=10):
    """
    Extracts the highest activating SAE features for a given text.
    SAFE: Returns empty list if sae is None.
    """
    # --- SAFETY CHECK ---
    if sae is None:
        return []
    # --------------------

    # 1. Get Model Activations
    # We use transformer_lens to get the residual stream at the specific layer
    # model can be the name or the object. If object, use it directly.
    if isinstance(model, str):
        # If passed string, we assume we can't get activations easily without the object
        # In this app, we should pass the ModelWrapper object.
        return []
        
    # Assuming 'model' is the ModelWrapper or HookedTransformer
    # We need the hook name for the residual stream
    hook_name = f"blocks.{layer}.hook_resid_post"
    
    # Run model with cache
    _, cache = model.run_with_cache(text, names_filter=[hook_name])
    activations = cache[hook_name][0, -1, :] # Last token, shape [d_model]

    # 2. Run SAE
    with torch.no_grad():
        # acts = sae.encode(activations)
        # Using forward because your previous code might have used it as a callable
        _, _, acts, _, _ = sae(activations)
    
    # 3. Get Top K
    values, indices = torch.topk(acts, top_k)
    
    features = []
    for val, idx in zip(values, indices):
        if val.item() > 0: # Only include firing features
            features.append({
                "feature_id": idx.item(),
                "activation": val.item()
            })
            
    return features

def get_feature_activations_for_text(text, model, layer, sae):
    """
    Returns the full feature vector for the last token.
    """
    if sae is None:
        return torch.tensor([])

    hook_name = f"blocks.{layer}.hook_resid_post"
    _, cache = model.run_with_cache(text, names_filter=[hook_name])
    activations = cache[hook_name][0, -1, :]
    
    with torch.no_grad():
        _, _, acts, _, _ = sae(activations)
        
    return acts

def get_sae_feature_vector(feature_idx, model_name, layer, sae_path=None):
    """
    Extracts the decoder direction (steering vector) for a specific feature.
    """
    sae = load_sae(model_name, layer, sae_path)
    if sae is None:
        return None
        
    # The feature direction is the column in W_dec
    # shape of W_dec is [d_hidden, d_input]
    # So we take the row feature_idx
    if feature_idx >= sae.W_dec.shape[0]:
        return None
        
    vector = sae.W_dec[feature_idx, :]
    return vector
