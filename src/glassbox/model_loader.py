import torch
import gc
from transformer_lens import HookedTransformer


class ModelWrapper:
    """
    Singleton wrapper with memory management.
    Allows switching models by clearing RAM first.
    """
    _instance = None
    _current_model_name = None

    @classmethod
    def load(cls, model_name="gpt2"):
        # If the requested model is ALREADY loaded, just return it.
        if cls._instance is not None and cls._current_model_name == model_name:
            return cls._instance

        # If a DIFFERENT model is loaded, we must clear it first to save RAM.
        if cls._instance is not None:
            print(f"🔄 Switching models: Unloading {cls._current_model_name}...")
            del cls._instance
            cls._instance = None
            gc.collect()  # Python Garbage Collection
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        print(f"⬇️ Loading {model_name} into RAM...")
        try:
            # Load the new model
            cls._instance = HookedTransformer.from_pretrained(model_name)
            cls._instance.eval()
            cls._current_model_name = model_name
            print(f"✅ Model {model_name} loaded successfully.")
        except Exception as e:
            print(f"❌ FAILED to load model: {model_name}")
            raise e

        return cls._instance