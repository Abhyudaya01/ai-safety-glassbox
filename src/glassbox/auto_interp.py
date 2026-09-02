import os
import torch
from glassbox.sae import get_feature_activations_for_text


def generate_feature_label(feature_idx, sae, model, layer_idx, corpus_path=None):
    # 1. Setup Gemini
    try:
        import google.generativeai as genai
    except ImportError:
        return "Error: install google-generativeai to use auto-labeling."

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: No API Key found."

    try:
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel('models/gemini-flash-latest')
    except Exception as e:
        return f"Error configuring Gemini: {e}"

    # 2. Find Activating Examples
    if not corpus_path:
        corpus_path = "data/sae_corpus.txt"

    examples = []

    print(f"\n--- DEBUG: Scanning Feature {feature_idx} ---")

    if os.path.exists(corpus_path):
        with open(corpus_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        pairs = []
        for line in lines:
            try:
                # --- FIX: Use model.cfg.model_name instead of model.model_name ---
                name_str = model.cfg.model_name if hasattr(model, "cfg") else "gpt2"

                vec = get_feature_activations_for_text(line, name_str, layer_idx, sae)

                if feature_idx < vec.shape[0]:
                    val = vec[feature_idx].item()

                    # PRINT THE EXACT VALUE TO TERMINAL
                    print(f"Sentence: '{line[:20]}...' -> Activation: {val:.4f}")

                    # THRESHOLD (Keep it low for testing)
                    if val > 0.001:
                        pairs.append((val, line))
            except Exception as e:
                print(f"Error processing line: {e}")
                continue

        # Sort and pick top 5
        pairs.sort(key=lambda x: x[0], reverse=True)
        examples = [p[1] for p in pairs[:5]]
        print(f"--- Found {len(examples)} activating examples ---\n")

    # 3. Construct Prompt
    if not examples:
        return "No activating sentences found. Check Terminal for values."

    prompt = f"""
    I am analyzing a neuron in an AI language model.
    It activates for these sentences:
    {examples}

    Provide a concise label (3-5 words) for the common concept.
    Format: "Label: [Your Label]"
    """

    # 4. Call API
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"API Error: {e}"
