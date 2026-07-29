import json
import os

# Load config.json once from the same directory as this file
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    CONFIG = json.load(_f)

def get_config():
    """Returns the loaded configuration dictionary."""
    return CONFIG

def get_llm_client_and_model():
    """
    Returns (OpenAI_client, model_name) based on the provider selected in config.json under 'llm'.
    Supports 'openrouter', 'openai', 'groq', 'gemini', or custom providers.
    """
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv(override=True)
    
    llm_config = CONFIG.get("llm", {})
    provider_name = llm_config.get("provider", "openrouter")
    providers_map = llm_config.get("providers", {})
    
    provider_info = providers_map.get(provider_name, {})
    model_name = provider_info.get("model", "meta-llama/llama-3-8b-instruct:free")
    base_url = provider_info.get("base_url")
    env_key_name = provider_info.get("env_key", "OPENROUTER_API_KEY")
    api_key = os.environ.get(env_key_name)
    
    if not api_key:
        print(f"Warning: Missing API key '{env_key_name}' for LLM provider '{provider_name}'.")
        
    client = OpenAI(api_key=api_key, base_url=base_url)
    return client, model_name

def get_device() -> str:
    """
    Returns 'cuda' if a GPU is available via PyTorch, otherwise 'cpu'.
    Centralized here so developers don't need to duplicate torch.cuda checks across modules.
    """
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

def clear_gpu_memory():
    """
    Clears PyTorch CUDA cache and runs garbage collection if CUDA is available.
    Use after processing heavy models (Whisper, CLIP, etc.) to keep VRAM clean.
    """
    try:
        import torch
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
