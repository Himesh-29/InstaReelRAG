import logging
import os
import glob
from datetime import datetime

_CURRENT_RUN_LOG_FILE = None

def _get_or_create_run_log_file() -> str:
    """
    Returns the log file path for the current execution run.
    Creates a new timestamped log file in the configured 'logs' directory per run
    and removes old log files if the count exceeds max_log_files.
    """
    global _CURRENT_RUN_LOG_FILE
    if _CURRENT_RUN_LOG_FILE is not None:
        return _CURRENT_RUN_LOG_FILE

    log_dir = "logs"
    max_log_files = 15
    try:
        from config import get_config
        cfg = get_config().get("logging", {})
        log_dir = cfg.get("log_dir", "logs")
        max_log_files = cfg.get("max_log_files", 15)
    except Exception:
        pass

    os.makedirs(log_dir, exist_ok=True)

    # Clean up older log files if we exceed (max_log_files - 1) before creating the new one
    try:
        existing_logs = sorted(
            glob.glob(os.path.join(log_dir, "*.log")),
            key=os.path.getmtime
        )
        # While existing logs count >= max_log_files, delete the oldest
        while len(existing_logs) >= max_log_files:
            oldest = existing_logs.pop(0)
            try:
                os.remove(oldest)
            except Exception:
                pass
    except Exception as e:
        print(f"Warning: Log rotation error: {e}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    _CURRENT_RUN_LOG_FILE = os.path.join(log_dir, f"run_{timestamp}.log")
    return _CURRENT_RUN_LOG_FILE

def setup_logger(name: str = "InstaReelRAG", log_file: str = None, level=logging.INFO) -> logging.Logger:
    """
    Sets up a logger that outputs messages to both:
    1. The terminal console
    2. A timestamped log file in the logs/ directory (one per run, with automatic rotation)
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding duplicate handlers if setup_logger is called multiple times
    if logger.handlers:
        return logger
        
    formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s", "%Y-%m-%d %H:%M:%S")
    
    # 1. Console Handler (prints to terminal)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 2. File Handler (saves to log file on disk)
    if log_file is None:
        log_file = _get_or_create_run_log_file()
        
    try:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not create log file {log_file}: {e}")
        
    return logger
