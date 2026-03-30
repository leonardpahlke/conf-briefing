"""Local speech-to-text transcription using OpenAI Whisper.

This module is intentionally decoupled from the rest of the project.
It only depends on openai-whisper (+ torch) and returns plain dicts.

Works with NVIDIA (CUDA), AMD (ROCm), and CPU.

Usage:
    from conf_briefing.collect.whisper_transcribe import transcribe
    result = transcribe("path/to/audio.mp3", model_name="base")
"""

from pathlib import Path


def _detect_device() -> str:
    """Auto-detect the best available compute device."""
    import torch

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"[whisper] Using GPU: {name}")
        return "cuda"

    print("[whisper] No GPU detected, using CPU (this will be slower)")
    return "cpu"


def transcribe(
    audio_path: str | Path,
    *,
    model_name: str = "base",
    language: str | None = None,
    device: str | None = None,
) -> dict:
    """Transcribe an audio file using Whisper.

    Args:
        audio_path: Path to audio file (mp3, wav, etc.)
        model_name: Whisper model size (tiny, base, small, medium, large-v3)
        language: Language code (e.g. "en"). None for auto-detection.
        device: Force device ("cuda", "cpu"). None for auto-detection.

    Returns:
        Dict with keys: segments, full_text, language, model
        Segment format matches youtube-transcript-api output:
        {"text": str, "start": float, "duration": float}
    """
    import whisper

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if device is None:
        device = _detect_device()

    print(f"[whisper] Loading model '{model_name}' on {device}...")
    model = whisper.load_model(model_name, device=device)

    print(f"[whisper] Transcribing {audio_path.name}...")
    result = model.transcribe(
        str(audio_path),
        language=language,
        verbose=False,
    )

    # Convert to our standard segment format
    segments = []
    for seg in result.get("segments", []):
        segments.append(
            {
                "text": seg["text"].strip(),
                "start": seg["start"],
                "duration": seg["end"] - seg["start"],
            }
        )

    full_text = " ".join(s["text"] for s in segments)
    detected_lang = result.get("language", language or "unknown")

    return {
        "segments": segments,
        "full_text": full_text,
        "language": detected_lang,
        "model": model_name,
    }


def transcribe_batch(
    audio_files: list[tuple[str, Path]],
    *,
    model_name: str = "base",
    language: str | None = None,
    device: str | None = None,
) -> list[tuple[str, dict | None]]:
    """Transcribe multiple audio files.

    Args:
        audio_files: List of (video_id, audio_path) tuples.
        model_name: Whisper model size.
        language: Language code or None for auto-detect.
        device: Force device or None for auto-detect.

    Returns:
        List of (video_id, transcript_dict | None) tuples.
    """
    import whisper

    if device is None:
        device = _detect_device()

    print(f"[whisper] Loading model '{model_name}' on {device}...")
    model = whisper.load_model(model_name, device=device)

    results = []
    total = len(audio_files)

    for i, (vid, audio_path) in enumerate(audio_files, 1):
        print(f"[whisper]   ({i}/{total}) {audio_path.name}...")
        try:
            raw = model.transcribe(str(audio_path), language=language, verbose=False)

            segments = [
                {
                    "text": seg["text"].strip(),
                    "start": seg["start"],
                    "duration": seg["end"] - seg["start"],
                }
                for seg in raw.get("segments", [])
            ]

            transcript = {
                "segments": segments,
                "full_text": " ".join(s["text"] for s in segments),
                "language": raw.get("language", language or "unknown"),
                "model": model_name,
            }
            results.append((vid, transcript))
        except Exception as e:
            print(f"[whisper]   ({i}/{total}) failed: {e}")
            results.append((vid, None))

    return results
