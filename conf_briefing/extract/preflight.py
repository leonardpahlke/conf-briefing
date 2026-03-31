"""Preflight checks for the extract pipeline."""

import importlib.util
import shutil
from dataclasses import dataclass
from pathlib import Path

from conf_briefing.config import Config
from conf_briefing.console import console, tag


@dataclass
class ExtractContext:
    """Resolved extract pipeline settings after auto-detection."""

    # Transcription
    transcribe_backend: str  # "whisper-cpp" or "faster-whisper"
    device: str  # "cpu", "cuda", "rocm"
    compute_type: str  # "int8", "float16"
    whisper_model: str  # model name (faster-whisper) or ggml path
    wcpp_binary: str | None  # path to whisper-cpp binary
    wcpp_model_path: str | None  # path to ggml model file
    initial_prompt: str  # domain-specific terminology

    # OCR
    ocr_backend: str  # "tesseract"


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


def _detect_rocm_gpu() -> bool:
    """Check if a ROCm GPU is available."""
    if Path("/dev/kfd").exists():
        return True
    if shutil.which("rocm-smi"):
        import subprocess

        result = subprocess.run(["rocm-smi", "--showid"], capture_output=True, text=True)
        return result.returncode == 0
    return False


def _find_wcpp_model(model_name: str) -> str | None:
    """Look for a ggml model file in standard cache locations."""
    cache_dirs = [
        Path.home() / ".cache" / "whisper-cpp",
        Path.home() / ".cache" / "whisper.cpp",
        Path("/usr/share/whisper.cpp/models"),
    ]

    # Map common model names to ggml filenames
    ggml_name = f"ggml-{model_name}.bin"

    for d in cache_dirs:
        model_path = d / ggml_name
        if model_path.exists():
            return str(model_path)

    return None


def _detect_transcription_backend(
    config: Config,
) -> tuple[str, str, str, str, str | None, str | None]:
    """Auto-detect best transcription backend.

    Returns (backend, device, compute_type, model, wcpp_binary, wcpp_model_path).
    """
    wcpp_bin = shutil.which("whisper-cpp")
    if wcpp_bin:
        wcpp_model = _find_wcpp_model(config.extract.whisper_model)
        if wcpp_model:
            device = "rocm" if _detect_rocm_gpu() else "cpu"
            console.print(f"{tag('preflight')} Transcription: whisper.cpp on {device.upper()}")
            return (
                "whisper-cpp",
                device,
                "float16",
                config.extract.whisper_model,
                wcpp_bin,
                wcpp_model,
            )
        else:
            console.print(
                f"{tag('preflight')} whisper-cpp found but no ggml model "
                f"for '{config.extract.whisper_model}' in cache. "
                f"Download with: whisper-cpp-download-ggml-model {config.extract.whisper_model}"
            )

    # Fallback to faster-whisper
    if importlib.util.find_spec("faster_whisper") is None:
        raise RuntimeError(
            "No transcription backend available. "
            "Install faster-whisper (uv sync --extra extract) or whisper-cpp."
        )

    device, compute_type = _resolve_device(config.extract.device, config.extract.compute_type)
    console.print(f"{tag('preflight')} Transcription: faster-whisper on {device} ({compute_type})")
    return (
        "faster-whisper",
        device,
        compute_type,
        config.extract.whisper_model,
        None,
        None,
    )


def _check_ocr_backend() -> None:
    """Check that Tesseract OCR is available."""
    if importlib.util.find_spec("pytesseract") is None:
        raise RuntimeError(
            "Missing required Python package: pytesseract. Install with: uv sync --extra extract"
        )


def _check_required_packages() -> None:
    """Check that required Python packages are installed and importable.

    Imports cv2 first (and verifies native bindings) since scenedetect
    depends on it and cv2 can silently half-load without its C extension.
    """
    # cv2 must be checked first — its native extension can fail silently,
    # leaving the module importable but without VideoCapture etc.
    _check_cv2()

    required = {"scenedetect": "scenedetect", "imagehash": "imagehash"}
    missing = []
    for module, pkg in required.items():
        if importlib.util.find_spec(module) is None:
            missing.append(pkg)
            continue
        try:
            importlib.import_module(module)
        except (ImportError, AttributeError) as e:
            raise RuntimeError(
                f"Package '{pkg}' is installed but failed to import: {e}\n"
                "This usually means a native library is missing. "
                "On NixOS/nix-shell, check LD_LIBRARY_PATH in flake.nix."
            ) from e
    if missing:
        raise RuntimeError(
            f"Missing required Python packages: {', '.join(missing)}. "
            "Install with: uv sync --extra extract"
        )


def _check_cv2() -> None:
    """Verify OpenCV is installed and its native C extension loads correctly."""
    if importlib.util.find_spec("cv2") is None:
        raise RuntimeError(
            "Missing required Python package: opencv-python. Install with: uv sync --extra extract"
        )
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError(
            f"opencv-python is installed but failed to import: {e}\n"
            "This usually means a native library is missing. "
            "On NixOS/nix-shell, check LD_LIBRARY_PATH in flake.nix."
        ) from e

    if not hasattr(cv2, "VideoCapture"):
        # Native extension failed silently — probe for the real error
        import os

        ld_path = os.environ.get("LD_LIBRARY_PATH", "(not set)")
        raise RuntimeError(
            "OpenCV loaded without native bindings (cv2.VideoCapture missing).\n"
            "The C extension likely failed to load due to a missing system library.\n"
            f"Current LD_LIBRARY_PATH: {ld_path}\n"
            "On NixOS/nix-shell, add the missing library to LD_LIBRARY_PATH in flake.nix."
        )


def _check_binaries(ocr_backend: str) -> None:
    """Check that required system binaries are on PATH."""
    errors = []
    if not shutil.which("ffmpeg"):
        errors.append("ffmpeg not found — add `pkgs.ffmpeg` to flake.nix or `apt install ffmpeg`")
    if ocr_backend == "tesseract" and not shutil.which("tesseract"):
        errors.append(
            "tesseract not found — add `pkgs.tesseract` to flake.nix or `apt install tesseract-ocr`"
        )
    if errors:
        raise RuntimeError("Missing system dependencies:\n  " + "\n  ".join(errors))


def _ensure_whisper_model(model_name: str) -> None:
    """Check if the faster-whisper model is cached; download if missing."""
    from faster_whisper.utils import download_model

    try:
        download_model(model_name, local_files_only=True)
        console.print(f"{tag('preflight')} Whisper model '{model_name}' cached locally.")
    except Exception:
        console.print(f"{tag('preflight')} Whisper model '{model_name}' not cached, downloading...")
        download_model(model_name)
        console.print(f"{tag('preflight')} Whisper model '{model_name}' downloaded.")


def _check_vlm_model(config: Config, vlm_model: str) -> None:
    """Verify Ollama is reachable and the VLM model is available."""
    try:
        import ollama

        client = ollama.Client(host=config.llm.ollama_base_url)
        client.show(vlm_model)
        console.print(f"{tag('preflight')} VLM: {vlm_model} (available)")
    except Exception as exc:
        console.print(
            f"{tag('preflight')} [yellow]VLM model '{vlm_model}' not available: {exc}. "
            f"Run 'ollama pull {vlm_model}' or clear vlm_model to skip.[/yellow]"
        )


def check_extract_ready(config: Config) -> ExtractContext:
    """Preflight checks for extract pipeline. Returns resolved ExtractContext."""
    console.print(f"{tag('preflight')} Running preflight checks...")

    _check_required_packages()
    console.print(f"{tag('preflight')} Required packages OK.")

    # Auto-detect transcription backend
    backend, device, compute_type, model, wcpp_bin, wcpp_model = _detect_transcription_backend(
        config
    )
    _check_ocr_backend()
    console.print(f"{tag('preflight')} OCR: Tesseract")

    # Check system binaries
    _check_binaries("tesseract")
    console.print(f"{tag('preflight')} System binaries OK.")

    # Check model availability
    if backend == "faster-whisper":
        _ensure_whisper_model(model)

    # VLM model check
    vlm_model = config.extract.vlm_model
    if vlm_model:
        _check_vlm_model(config, vlm_model)

    console.print(f"{tag('preflight')} All checks passed.")
    return ExtractContext(
        transcribe_backend=backend,
        device=device,
        compute_type=compute_type,
        whisper_model=model,
        wcpp_binary=wcpp_bin,
        wcpp_model_path=wcpp_model,
        initial_prompt=config.extract.initial_prompt,
        ocr_backend="tesseract",
    )
