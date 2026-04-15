"""Preflight checks for the extract pipeline."""

import importlib.util
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from conf_briefing.config import Config
from conf_briefing.console import console, tag


@dataclass
class ExtractContext:
    """Resolved extract pipeline settings after auto-detection."""

    # Transcription
    transcribe_backend: str  # "whisper-cpp", "faster-whisper", or "whisperx"
    device: str  # "cpu", "cuda", "rocm"
    compute_type: str  # "int8", "float16"
    whisper_model: str  # model name (faster-whisper/whisperx) or ggml path
    wcpp_binary: str | None  # path to whisper-cpp binary
    wcpp_model_path: str | None  # path to ggml model file
    initial_prompt: str  # domain-specific terminology
    diarize: bool  # enable speaker diarization (whisperx only)

    # OCR
    ocr_backend: str  # "tesseract"


def _gpu_sanity_check() -> bool:
    """Run a small GPU tensor op in a subprocess to verify the GPU actually works.

    Returns True if GPU operations succeed, False if they segfault or error.
    This catches unsupported GPU architectures (e.g. RDNA4 on ROCm 6.2).
    """
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch; t = torch.zeros(1, device='cuda'); print(float(t))",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.returncode == 0


def _detect_gpu() -> tuple[str, str]:
    """Detect GPU availability with stderr suppression for noisy runtimes.

    Returns (device, gpu_name) where device is "cpu", "cuda", or "rocm".
    """
    try:
        import contextlib
        import os
        import sys

        # ROCm workaround: ctranslate2's native allocator crashes with
        # "free(): invalid pointer" if torch's ROCm/HIP runtime initializes
        # first. Import ctranslate2 before torch to avoid this.
        with contextlib.suppress(ImportError):
            import ctranslate2 as _  # noqa: F401
        import torch

        _stderr_fd = sys.stderr.fileno()
        _old_stderr_fd = os.dup(_stderr_fd)
        with open(os.devnull, "w") as _devnull:
            os.dup2(_devnull.fileno(), _stderr_fd)
            try:
                cuda_available = torch.cuda.is_available()
            finally:
                os.dup2(_old_stderr_fd, _stderr_fd)
                os.close(_old_stderr_fd)

        if not cuda_available:
            return "cpu", ""

        # ROCm torch exposes GPUs via torch.cuda but sets torch.version.hip
        is_rocm = bool(getattr(torch.version, "hip", None))
        gpu_name = torch.cuda.get_device_name(0)

        # Verify GPU actually works — catches unsupported architectures
        # (e.g. RDNA4/gfx1151 on ROCm 6.2 which only supports RDNA3).
        if _gpu_sanity_check():
            device = "rocm" if is_rocm else "cuda"
            return device, gpu_name

        return "cpu", gpu_name

    except ImportError:
        return "cpu", ""


def _resolve_device(config_device: str, config_compute: str) -> tuple[str, str]:
    """Resolve device and compute_type from config ('auto' -> detect GPU).

    Returns (device, compute_type) where device is one of:
      "cpu"   — no GPU available
      "cuda"  — NVIDIA CUDA GPU
      "rocm"  — AMD ROCm GPU (torch.cuda works via HIP, but ctranslate2 does not)
    """
    device = config_device
    compute_type = config_compute

    if device == "auto":
        device, gpu_name = _detect_gpu()
        if device in ("cuda", "rocm"):
            console.print(
                f"{tag('preflight')} GPU: {gpu_name} ({'ROCm' if device == 'rocm' else 'CUDA'})"
            )
        elif gpu_name:
            console.print(
                f"{tag('preflight')} [yellow]GPU detected ({gpu_name}) "
                f"but operations failed — falling back to CPU. "
                f"Your GPU may not be supported by this ROCm/CUDA version.[/yellow]"
            )

    if compute_type == "auto":
        compute_type = "float16" if device in ("cuda", "rocm") else "int8"

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


class TranscriptionBackend(NamedTuple):
    backend: str
    device: str
    compute_type: str
    model: str
    wcpp_binary: str | None
    wcpp_model_path: str | None


def _detect_transcription_backend(
    config: Config,
) -> TranscriptionBackend:
    """Auto-detect best transcription backend.

    Priority: whisperx (diarization) > whisper-cpp (ROCm) > faster-whisper.
    """
    # Prefer WhisperX if installed (batched inference, VAD, alignment, diarization).
    # Diarization gracefully degrades without HF_TOKEN.
    if importlib.util.find_spec("whisperx") is not None:
        device, compute_type = _resolve_device(config.extract.device, config.extract.compute_type)
        hf_token = os.environ.get("HF_TOKEN", "")
        can_diarize = config.extract.diarize and bool(hf_token)
        diarize_note = ""
        if config.extract.diarize and not hf_token:
            diarize_note = " [yellow](set HF_TOKEN to enable diarization)[/yellow]"
        elif can_diarize:
            diarize_note = " with diarization"
        device_desc = (
            "rocm (ASR on CPU, align/diarize on GPU)"
            if device == "rocm"
            else f"{device} ({compute_type})"
        )
        console.print(f"{tag('preflight')} Transcription: whisperx on {device_desc}{diarize_note}")
        return TranscriptionBackend(
            "whisperx",
            device,
            compute_type,
            config.extract.whisper_model,
            None,
            None,
        )

    wcpp_bin = shutil.which("whisper-cpp")
    if wcpp_bin:
        wcpp_model = _find_wcpp_model(config.extract.whisper_model)
        if wcpp_model:
            device = "rocm" if _detect_rocm_gpu() else "cpu"
            console.print(f"{tag('preflight')} Transcription: whisper.cpp on {device.upper()}")
            return TranscriptionBackend(
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
    return TranscriptionBackend(
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


_PYANNOTE_GATED_MODELS = [
    "pyannote/speaker-diarization-3.1",
    "pyannote/segmentation-3.0",
]


def _check_hf_token_and_diarize_access(hf_token: str) -> None:
    """Verify HF_TOKEN is set, valid, and has access to gated pyannote models."""
    from huggingface_hub import whoami
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    # 1. Token set?
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN is not set. Diarization requires a HuggingFace token.\n"
            "  1. Create a token at https://hf.co/settings/tokens (Read access)\n"
            "  2. Add HF_TOKEN=hf_... to your .env file"
        )

    # 2. Token valid?
    try:
        user_info = whoami(token=hf_token)
        username = user_info.get("name", "unknown")
    except Exception as exc:
        raise RuntimeError(
            f"HF_TOKEN is invalid: {exc}\n  Create a new token at https://hf.co/settings/tokens"
        ) from exc

    # 3. Gated model access?
    from huggingface_hub import auth_check

    for repo_id in _PYANNOTE_GATED_MODELS:
        try:
            auth_check(repo_id, token=hf_token)
        except GatedRepoError as exc:
            raise RuntimeError(
                f"HF user '{username}' has not accepted the license for '{repo_id}'.\n"
                f"  Visit https://hf.co/{repo_id} and click 'Agree and access repository'."
            ) from exc
        except RepositoryNotFoundError as exc:
            raise RuntimeError(
                f"Model '{repo_id}' not found. Check your HF_TOKEN permissions.\n"
                "  The token needs: Read access to contents of all public gated repos."
            ) from exc
        except Exception as exc:
            console.print(
                f"{tag('preflight')} [yellow]Could not verify access to '{repo_id}': {exc}[/yellow]"
            )

    console.print(
        f"{tag('preflight')} HF token valid (user: {username}), pyannote model access OK."
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

    # Diarization checks
    if config.extract.diarize and backend != "whisperx":
        console.print(
            f"{tag('preflight')} [yellow]Diarization requires whisperx. "
            f"Install with: just pull-models <event> amd|nvidia[/yellow]"
        )
    elif config.extract.diarize and backend == "whisperx":
        _check_hf_token_and_diarize_access(os.environ.get("HF_TOKEN", ""))

    console.print(f"{tag('preflight')} All checks passed.")
    return ExtractContext(
        transcribe_backend=backend,
        device=device,
        compute_type=compute_type,
        whisper_model=model,
        wcpp_binary=wcpp_bin,
        wcpp_model_path=wcpp_model,
        initial_prompt=config.extract.initial_prompt,
        diarize=config.extract.diarize,
        ocr_backend="tesseract",
    )
