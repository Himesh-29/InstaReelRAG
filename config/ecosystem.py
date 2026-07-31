import os
import sys
import subprocess
import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger("Ecosystem")

class EcosystemStrategy(ABC):
    """
    Abstract Base Class defining the utility interface for hardware and OS ecosystem operations.
    Follows the Factory Object Pattern to encapsulate platform-specific differences
    (macOS Apple Silicon MPS, Windows/Linux NVIDIA CUDA, or CPU-only).
    """
    @property
    @abstractmethod
    def name(self) -> str:
        """Returns a human-readable identifier for this ecosystem (e.g., 'mac_mps', 'windows_cuda')."""
        pass

    @property
    @abstractmethod
    def device(self) -> str:
        """Returns the PyTorch device string ('cuda', 'mps', or 'cpu')."""
        pass

    @abstractmethod
    def clear_gpu_cache(self) -> None:
        """Safely releases unused GPU VRAM memory cache for this ecosystem."""
        pass

    def run_command(self, cmd: list[str], check: bool = False) -> subprocess.CompletedProcess:
        """
        Executes a subprocess command optimized for the underlying OS ecosystem.
        Handles Windows cmd.exe vs POSIX shell syntax variations safely.
        """
        try:
            return subprocess.run(cmd, check=check)
        except Exception as e:
            logger.error(f"Command execution failed on ecosystem '{self.name}': {e}")
            raise

    def execute_ffmpeg_audio_extraction(self, video_path: str, output_audio_path: str, ffmpeg_exe: str = "ffmpeg") -> bool:
        """
        Executes ffmpeg audio extraction with OS-specific return code handling
        (handling Windows unsigned integer overflow return codes vs POSIX macOS/Linux return codes).
        """
        logger.info(f"[{self.name}] Extracting audio from '{video_path}' to '{output_audio_path}'...")
        cmd = [
            ffmpeg_exe,
            "-i", video_path,
            "-q:a", "0",
            "-map", "a",
            output_audio_path,
            "-y",
            "-loglevel", "error"
        ]
        try:
            res = self.run_command(cmd, check=False)
            if res.returncode != 0:
                # Windows unsigned integer overflow for negative return codes (4294967274 is -22 / EINVAL, no audio stream)
                # POSIX macOS/Linux returns negative or standard error codes for silent video streams
                if res.returncode in [4294967274, -22, 1]:
                    logger.info(f"[{self.name}] Video contains no audio stream (silent Reel). Skipping audio extraction.")
                else:
                    logger.error(f"[{self.name}] ffmpeg audio extraction failed with return code {res.returncode}")
                return False
            logger.info(f"[{self.name}] Audio extraction successful.")
            return True
        except Exception as e:
            logger.error(f"[{self.name}] Failed to extract audio with ffmpeg: {e}")
            return False

class CudaEcosystem(EcosystemStrategy):
    """Strategy for Windows or Linux systems with an NVIDIA CUDA-enabled GPU."""
    @property
    def name(self) -> str:
        return f"{sys.platform}_cuda"

    @property
    def device(self) -> str:
        return "cuda"

    def clear_gpu_cache(self) -> None:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.debug(f"Could not clear CUDA cache: {e}")

class MpsEcosystem(EcosystemStrategy):
    """Strategy for macOS systems with Apple Silicon Metal Performance Shaders (MPS)."""
    @property
    def name(self) -> str:
        return "mac_mps"

    @property
    def device(self) -> str:
        return "mps"

    def clear_gpu_cache(self) -> None:
        try:
            import torch
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except Exception as e:
            logger.debug(f"Could not clear MPS cache: {e}")

class CpuEcosystem(EcosystemStrategy):
    """Strategy for Windows, macOS, or Linux systems running on CPU without GPU acceleration."""
    @property
    def name(self) -> str:
        return f"{sys.platform}_cpu"

    @property
    def device(self) -> str:
        return "cpu"

    def clear_gpu_cache(self) -> None:
        # No-op on CPU-only systems
        pass

_INSTANCE: Optional[EcosystemStrategy] = None

def get_ecosystem() -> EcosystemStrategy:
    """
    Factory Method that inspects the current operating system and available hardware acceleration,
    instantiating and returning the appropriate EcosystemStrategy singleton.
    """
    global _INSTANCE
    if _INSTANCE is not None:
        return _INSTANCE

    try:
        import torch
        if torch.cuda.is_available():
            _INSTANCE = CudaEcosystem()
            logger.info(f"Ecosystem detected: NVIDIA CUDA ({_INSTANCE.name})")
            return _INSTANCE
        elif sys.platform == "darwin" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            _INSTANCE = MpsEcosystem()
            logger.info(f"Ecosystem detected: Apple Silicon Metal/MPS ({_INSTANCE.name})")
            return _INSTANCE
    except ImportError:
        pass

    _INSTANCE = CpuEcosystem()
    logger.info(f"Ecosystem detected: CPU-only ({_INSTANCE.name})")
    return _INSTANCE
