import torch
from .model_loader import ModelWrapper


def get_steering_vector(positive_prompt, negative_prompt, model_name, layer_idx):
    """
    Calculates a vector direction by subtracting two opposite concepts.
    Example: Vector = ("Love") - ("Hate")
    """
    model = ModelWrapper.load(model_name)

    # 1. Run both prompts and cache their internal states
    # We use run_with_cache to snag the activations at the specific layer
    _, cache_pos = model.run_with_cache(positive_prompt, remove_batch_dim=True)
    _, cache_neg = model.run_with_cache(negative_prompt, remove_batch_dim=True)

    # 2. Extract the residual stream at the target layer
    # We target the LAST token because that's where the prediction happens
    hook_name = f"blocks.{layer_idx}.hook_resid_post"
    pos_vec = cache_pos[hook_name][-1]
    neg_vec = cache_neg[hook_name][-1]

    # 3. Calculate the difference (The "Steering Vector")
    steering_vec = pos_vec - neg_vec
    return steering_vec


def generate_steered_response(input_prompt, steering_vec, multiplier, model_name, layer_idx, max_new_tokens=15):
    """
    Generates text while actively injecting the steering vector into the model.
    """
    model = ModelWrapper.load(model_name)

    # Define the "Hook" - this function runs INSIDE the model at every step
    def steering_hook(resid_post, hook):
        # Add the vector to the current state (Broadcasting across the sequence)
        # resid_post shape: [batch, seq_len, d_model]
        resid_post += steering_vec * multiplier
        return resid_post

    # Register the hook temporarily for this generation
    hook_name = f"blocks.{layer_idx}.hook_resid_post"

    # Run generation with the hook active
    with model.hooks(fwd_hooks=[(hook_name, steering_hook)]):
        output = model.generate(
            input_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Deterministic (so we know the change is real)
            verbose=False
        )

    return output