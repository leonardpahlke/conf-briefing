"""Video transcription using faster-whisper."""

import json
import logging
import time
from pathlib import Path

from conf_briefing.config import Config
from conf_briefing.console import console, tag


def transcribe_video(
    video_path: Path,
    output_dir: Path,
    model,
    model_name: str,
    initial_prompt: str = "",
) -> Path:
    """Transcribe a single video. Returns path to transcript JSON."""
    segments_iter, info = model.transcribe(
        str(video_path),
        beam_size=5,
        initial_prompt=initial_prompt or None,
    )

    segments = []
    full_parts = []
    for seg in segments_iter:
        segments.append({"start": round(seg.start, 2), "end": round(seg.end, 2), "text": seg.text})
        full_parts.append(seg.text)

    video_id = video_path.stem
    transcript = {
        "video_id": video_id,
        "title": "",
        "language": info.language,
        "duration_sec": round(info.duration, 1),
        "model": model_name,
        "full_text": " ".join(full_parts),
        "segments": segments,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{video_id}.json"
    out_path.write_text(json.dumps(transcript, ensure_ascii=False, indent=2))
    return out_path


def transcribe_all(
    config: Config,
    device: str,
    compute_type: str,
    initial_prompt: str = "",
) -> list[Path]:
    """Transcribe all downloaded videos. Returns list of transcript file paths."""
    data_dir = config.data_dir
    videos_dir = data_dir / "videos"
    output_dir = data_dir / "transcripts"

    if not videos_dir.exists():
        console.print(f"{tag('whisper')} No videos directory found, skipping transcription.")
        return []

    video_files = sorted(videos_dir.glob("*.mp4"))
    if not video_files:
        console.print(f"{tag('whisper')} No video files found.")
        return []

    # Filter out already-transcribed videos
    to_process = []
    existing = []
    for vf in video_files:
        transcript_path = output_dir / f"{vf.stem}.json"
        if transcript_path.exists():
            existing.append(transcript_path)
        else:
            to_process.append(vf)

    if existing:
        console.print(f"{tag('whisper')} Skipping {len(existing)} already-transcribed video(s).")

    if not to_process:
        console.print(f"{tag('whisper')} All videos already transcribed.")
        return existing

    model_name = config.extract.whisper_model

    console.print(
        f"{tag('whisper')} Transcribing {len(to_process)} video(s) "
        f"with {model_name} on {device} ({compute_type})."
    )

    # Suppress HuggingFace Hub token warning — model is already cached locally
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

    from faster_whisper import WhisperModel

    with console.status(f"{tag('whisper')} Loading model {model_name}..."):
        t0 = time.monotonic()
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        elapsed = time.monotonic() - t0
    console.print(f"{tag('whisper')} Model loaded in {elapsed:.1f}s.")

    results = list(existing)
    total = len(to_process)
    for i, vf in enumerate(to_process, 1):
        with console.status(f"{tag('whisper')} [{i}/{total}] Transcribing {vf.name}..."):
            t0 = time.monotonic()
            out = transcribe_video(vf, output_dir, model, model_name, initial_prompt)
            elapsed = time.monotonic() - t0
        console.print(f"{tag('whisper')} [{i}/{total}] {vf.name} ({elapsed:.0f}s)")
        results.append(out)

    console.print(f"{tag('whisper')} Transcribed {total} video(s).")
    return results
