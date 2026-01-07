import torch
import pandas as pd
import circuitsvis as cv
from .model_loader import ModelWrapper


def get_logit_lens_data(text, model_name):
    """
    Runs the model and intercepts the 'thought process' at every layer.
    Returns a DataFrame showing the top guess for the next word at each layer.
    """
    model = ModelWrapper.load(model_name)

    # Run model and cache activations
    with torch.no_grad():
        logits, cache = model.run_with_cache(text, remove_batch_dim=True)

    results = []
    n_layers = model.cfg.n_layers

    # Iterate through every layer to see what it was thinking
    for layer in range(n_layers):
        # Hook location: The end of the layer (after Attention + MLP)
        hook_name = f"blocks.{layer}.hook_resid_post"

        # Grab the residual stream (hidden state)
        resid = cache[hook_name]  # Shape: [seq_len, d_model]

        # We only care about the LAST token (the one predicting the next word)
        last_token_resid = resid[-1]

        # Decode the hidden state into a word
        # 1. Apply Layer Norm (simulating the final output stage)
        scaled_resid = model.ln_final(last_token_resid)
        # 2. Project to Vocabulary (Unembed)
        decoded = model.unembed(scaled_resid)
        # 3. Convert to Probability
        probs = torch.softmax(decoded, dim=-1)

        # Find the winner (Top 1)
        max_val, max_idx = torch.max(probs, dim=-1)
        predicted_token = model.to_string(max_idx)
        confidence = max_val.item()

        results.append({
            "Layer": layer,
            "Top Guess": predicted_token,
            "Confidence": f"{confidence:.1%}",
            "Top Prob": confidence
        })

    return pd.DataFrame(results)


def get_attention_data(text, model_name, layer=0):
    """
    Visualizes the Attention Heads for a specific layer.
    Shows which words 'look at' which other words.
    """
    model = ModelWrapper.load(model_name)

    # Get tokens as strings for the visualization
    str_tokens = model.to_str_tokens(text)

    # Run model to get attention patterns
    with torch.no_grad():
        _, cache = model.run_with_cache(text, remove_batch_dim=True)

    # Extract attention pattern for the specified layer
    # Shape: [num_heads, seq_len, seq_len]
    attention_pattern = cache[f"blocks.{layer}.attn.hook_pattern"]

    # generate the interactive HTML visualization
    html_object = cv.attention.attention_patterns(
        tokens=str_tokens,
        attention=attention_pattern
    )

    return str(html_object)