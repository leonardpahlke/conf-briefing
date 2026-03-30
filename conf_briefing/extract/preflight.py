"""Preflight checks for the extract pipeline."""

import shutil

from conf_briefing.config import Config
from conf_briefing.console import console, tag


def _resolve_device(config_device: str, config_compute: str) -> tuple[str, str]:
    """Resolve device and compute_type from config ('auto' -> detect CUDA)."""
    device = config_device
    compute_type = config_compute

    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"

    return device, compute_type


def _check_packages() -> None:
    """Check that required Python packages are importable."""
    missing = []
    for pkg in ("faster_whisper", "scenedetect", "cv2", "pytesseract", "imagehash"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        raise RuntimeError(
            f"Missing Python packages: {', '.join(missing)}. Install with: uv sync --extra extract"
        )


def _check_binaries() -> None:
    """Check that required system binaries are on PATH."""
    errors = []
    if not shutil.which("ffmpeg"):
        errors.append("ffmpeg not found — add `pkgs.ffmpeg` to flake.nix or `apt install ffmpeg`")
    if not shutil.which("tesseract"):
        errors.append(
            "tesseract not found — add `pkgs.tesseract` to flake.nix or `apt install tesseract-ocr`"
        )
    if errors:
        raise RuntimeError("Missing system dependencies:\n  " + "\n  ".join(errors))


def _ensure_whisper_model(model_name: str) -> None:
    """Check if the Whisper model is cached; download if missing."""
    from faster_whisper.utils import download_model

    try:
        download_model(model_name, local_files_only=True)
        console.print(f"{tag('preflight')} Whisper model '{model_name}' found in cache.")
    except Exception:
        console.print(f"{tag('preflight')} Whisper model '{model_name}' not cached, downloading...")
        download_model(model_name)
        console.print(f"{tag('preflight')} Whisper model '{model_name}' downloaded.")


def check_extract_ready(config: Config) -> tuple[str, str]:
    """Preflight checks for extract pipeline. Returns (device, compute_type)."""
    console.print(f"{tag('preflight')} Running preflight checks...")

    _check_packages()
    console.print(f"{tag('preflight')} Python packages OK.")

    _check_binaries()
    console.print(f"{tag('preflight')} System binaries OK (ffmpeg, tesseract).")

    _ensure_whisper_model(config.extract.whisper_model)

    device, compute_type = _resolve_device(config.extract.device, config.extract.compute_type)
    console.print(f"{tag('preflight')} Using {device} with {compute_type} quantization.")

    console.print(f"{tag('preflight')} All checks passed.")
    return device, compute_type
