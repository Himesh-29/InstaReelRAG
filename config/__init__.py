from dotenv import load_dotenv
load_dotenv(override=True)

from .config import get_config, CONFIG, get_llm_client_and_model, get_llm_chat_completion
from .logger import setup_logger
from .ecosystem import get_ecosystem

ECOSYSTEM = get_ecosystem()
DEVICE = ECOSYSTEM.device

__all__ = [
    "get_config",
    "CONFIG",
    "get_llm_client_and_model",
    "get_llm_chat_completion",
    "setup_logger",
    "ECOSYSTEM",
    "DEVICE",
    "get_ecosystem"
]

