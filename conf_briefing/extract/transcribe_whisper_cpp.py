"""Video transcription using whisper.cpp subprocess backend."""

import json
import subprocess
import tempfile
import time
from pathlib import Path

from conf_briefing.config import Config
from conf_briefing.console import console, tag
from conf_briefing.extract.transcribe import _load_video_title, _partition_videos
from conf_briefing.io import load_json_file


def _convert_to_wav(video_path: Path, wav_path: Path) -> None:
    """Convert video to 16kHz mono WAV for whisper.cpp."""
    cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-f",
        "wav",
        "-y",
        str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr[:500]}")


def _parse_wcpp_output(wcpp_json: dict, video_id: str, model_name: str, title: str = "") -> dict:
    """Convert whisper.cpp JSON output to our transcript schema."""
    segments = []
    full_parts = []

    for entry in wcpp_json.get("transcription", []):
        timestamps = entry.get("timestamps", {})
        start_str = timestamps.get("from", "00:00:00.000")
        end_str = timestamps.get("to", "00:00:00.000")
        text = entry.get("text", "").strip()

        start_sec = _timestamp_to_seconds(start_str)
        end_sec = _timestamp_to_seconds(end_str)

        segments.append(
            {
                "start": round(start_sec, 2),
                "end": round(end_sec, 2),
                "text": text,
            }
        )
        full_parts.append(text)

    duration = segments[-1]["end"] if segments else 0.0
    language = wcpp_json.get("result", {}).get("language", "en")

    return {
        "video_id": video_id,
        "title": title,
        "language": language,
        "duration_sec": round(duration, 1),
        "model": model_name,
        "full_text": " ".join(full_parts),
        "segments": segments,
    }


def _timestamp_to_seconds(ts: str) -> float:
    """Parse 'HH:MM:SS.mmm' or 'HH:MM:SS,mmm' to seconds."""
    ts = ts.replace(",", ".")
    parts = ts.split(":")
    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    return 0.0


def transcribe_video_wcpp(
    video_path: Path,
    output_dir: Path,
    model_path: str,
    wcpp_binary: str,
    initial_prompt: str = "",
) -> Path:
    """Transcribe a single video using whisper.cpp CLI. Returns path to transcript JSON."""
    video_id = video_path.stem

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / f"{video_id}.wav"
        json_prefix = Path(tmp) / video_id

        # Convert to WAV
        _convert_to_wav(video_path, wav_path)

        # Run whisper.cpp
        cmd = [
            wcpp_binary,
            "-m",
            model_path,
            "-f",
            str(wav_path),
            "-oj",  # JSON output
            "-of",
            str(json_prefix),  # output file prefix
        ]
        if initial_prompt:
            cmd.extend(["--prompt", initial_prompt])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            raise RuntimeError(f"whisper-cpp failed: {result.stderr[:500]}")

        # whisper.cpp writes <prefix>.json
        wcpp_json_path = Path(f"{json_prefix}.json")
        if not wcpp_json_path.exists():
            raise RuntimeError(f"whisper-cpp did not produce output at {wcpp_json_path}")

        wcpp_data = load_json_file(wcpp_json_path)

    title = _load_video_title(video_path)

    # Parse into our schema
    model_name = Path(model_path).stem
    transcript = _parse_wcpp_output(wcpp_data, video_id, model_name, title=title)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{video_id}.json"
    out_path.write_text(json.dumps(transcript, ensure_ascii=False, indent=2))
    return out_path


def transcribe_all_wcpp(
    config: Config,
    model_path: str,
    wcpp_binary: str,
    initial_prompt: str = "",
) -> list[Path]:
    """Transcribe all videos using whisper.cpp. Same skip-existing logic."""
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

    to_process, existing = _partition_videos(video_files, output_dir)

    if existing:
        console.print(f"{tag('whisper')} Skipping {len(existing)} already-transcribed video(s).")

    if not to_process:
        console.print(f"{tag('whisper')} All videos already transcribed.")
        return existing

    console.print(
        f"{tag('whisper')} Transcribing {len(to_process)} video(s) "
        f"with whisper.cpp ({Path(model_path).stem})."
    )

    results = list(existing)
    total = len(to_process)
    for i, vf in enumerate(to_process, 1):
        with console.status(f"{tag('whisper')} [{i}/{total}] Transcribing {vf.name}..."):
            t0 = time.monotonic()
            out = transcribe_video_wcpp(vf, output_dir, model_path, wcpp_binary, initial_prompt)
            elapsed = time.monotonic() - t0
        console.print(f"{tag('whisper')} [{i}/{total}] {vf.name} ({elapsed:.0f}s)")
        results.append(out)

    console.print(f"{tag('whisper')} Transcribed {total} video(s).")
    return results
