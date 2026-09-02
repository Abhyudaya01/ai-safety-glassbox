import torch
from glassbox.model_loader import ModelWrapper


def get_steering_vector(pos_text, neg_text, model_name, layer_idx):
    """
    Calculates a steering vector by subtracting the activations of a negative concept
    from a positive concept (e.g., "Love" - "Hate").
    """
    model = ModelWrapper.load(model_name)

    # 1. Run both prompts
    with torch.no_grad():
        _, cache_pos = model.run_with_cache(pos_text)
        _, cache_neg = model.run_with_cache(neg_text)

    # 2. Extract activations at the target layer
    hook_point = f"blocks.{layer_idx}.hook_resid_pre"

    # Get the activation of the *last token* (the most context-heavy)
    # Shape: [batch, pos, d_model] -> [1, pos, d_model]
    if hook_point in cache_pos:
        act_pos = cache_pos[hook_point][0, -1, :]
        act_neg = cache_neg[hook_point][0, -1, :]
    else:
        raise ValueError(f"Layer {layer_idx} not found in model cache.")

    # 3. Calculate Direction (Pos - Neg)
    steering_vec = act_pos - act_neg

    # Normalize the vector (unit length) so 'strength' slider controls magnitude explicitly
    steering_vec = steering_vec / (torch.norm(steering_vec) + 1e-5)

    return steering_vec


def run_steering_eval(prompt, vector, multiplier, model_name, layer_idx, max_new_tokens=40):
    """
    Generates two texts:
    1. Control (No steering)
    2. Steered (With the vector added at the specific layer)
    """
    model = ModelWrapper.load(model_name)
    hook_point = f"blocks.{layer_idx}.hook_resid_pre"

    # --- 1. Control Generation (Normal) ---
    control_output = model.generate(prompt, max_new_tokens=max_new_tokens, verbose=False)

    # --- 2. Steered Generation (Hooked) ---

    # Define the hook function
    def steering_hook(resid_pre, hook):
        # resid_pre shape: [batch, pos, d_model]
        # vector shape: [d_model]

        # Ensure vector is on the same device (GPU/CPU)
        vec_device = vector.to(resid_pre.device)

        # Add vector to ALL token positions (broadcasting)
        return resid_pre + (vec_device * multiplier)

    # Run with the hook active
    with model.hooks(fwd_hooks=[(hook_point, steering_hook)]):
        steered_output = model.generate(prompt, max_new_tokens=max_new_tokens, verbose=False)

    return {
        "control_text": control_output,
        "steered_text": steered_output,
        # Calculate simple metrics for display
        "metrics": {
            "sentiment_steered": 0.0,
            "sentiment_delta": 0.0,
            "subjectivity_steered": 0.0,
            "subjectivity_control": 0.0
        }
    }


def generate_steered_response(input_prompt, steering_vec, multiplier, model_name, layer_idx, max_new_tokens=15):
    """
    Backward-compatible helper used by evaluate.py.
    """
    return run_steering_eval(
        input_prompt,
        steering_vec,
        multiplier,
        model_name,
        layer_idx,
        max_new_tokens=max_new_tokens,
    )["steered_text"]
