import circuitsvis as cv
from .model_loader import ModelWrapper

def get_attention_pattern(text_input, layer_idx, model_name="gpt2"): # <--- CHANGED THIS
    model = ModelWrapper.load(model_name)
    logits, cache = model.run_with_cache(text_input, remove_batch_dim=True)
    attn_name = f"blocks.{layer_idx}.attn.hook_pattern"
    attention_pattern = cache[attn_name]
    str_tokens = model.to_str_tokens(text_input)
    return str_tokens, attention_pattern

def generate_heatmap_html(text_input, layer_idx, model_name="gpt2"): # <--- CHANGED THIS
    tokens, attn = get_attention_pattern(text_input, layer_idx, model_name)
    html_obj = cv.attention.attention_patterns(tokens=tokens, attention=attn)
    return str(html_obj)