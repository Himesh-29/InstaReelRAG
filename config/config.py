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

def get_llm_chat_completion(messages: list[dict], temperature: float = None, max_tokens: int = None) -> str:
    """
    Unified helper function that invokes the configured LLM provider and returns the message content.
    Encapsulates client instantiation, model selection, and default parameter fallback.
    """
    llm_config = CONFIG.get("llm", {})
    if temperature is None:
        temperature = llm_config.get("temperature", 0.1)
    if max_tokens is None:
        max_tokens = llm_config.get("max_tokens", 512)

    client, model_name = get_llm_client_and_model()
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

