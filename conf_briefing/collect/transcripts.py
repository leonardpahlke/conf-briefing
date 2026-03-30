"""Fetch or generate transcripts for conference recordings.

Strategies:
  - "api": Use youtube-transcript-api for YouTube subtitles (fast, may fail)
  - "local": Download audio via yt-dlp, transcribe with Whisper (any language, no limits)
"""

import json
import re
from pathlib import Path

from conf_briefing.config import Config


def extract_video_ids_from_playlist(playlist_url: str) -> list[str]:
    """Extract video IDs from a YouTube playlist URL.

    Uses requests to fetch the playlist page and extract video IDs from the HTML.
    """
    import requests

    resp = requests.get(playlist_url, timeout=30)
    resp.raise_for_status()
    # Extract video IDs from the playlist page HTML
    ids = re.findall(r'"videoId":"([a-zA-Z0-9_-]{11})"', resp.text)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for vid in ids:
        if vid not in seen:
            seen.add(vid)
            unique.append(vid)
    return unique


def _validate_video_id(vid: str) -> str:
    """Validate a YouTube video ID."""
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,20}", vid):
        raise ValueError(f"Invalid video ID: {vid!r}")
    return vid


def _collect_video_ids(config: Config) -> list[str]:
    """Gather all video IDs from config (direct IDs + playlist)."""
    video_ids = [_validate_video_id(v) for v in config.conference.recordings.video_ids]

    if config.conference.recordings.youtube_playlist:
        print("[collect] Extracting video IDs from playlist...")
        playlist_ids = extract_video_ids_from_playlist(
            config.conference.recordings.youtube_playlist
        )
        print(f"[collect] Found {len(playlist_ids)} videos in playlist.")
        video_ids.extend(playlist_ids)

    return video_ids


# -- API strategy (youtube-transcript-api) ------------------------------------


def _fetch_transcript_api(video_id: str) -> dict:
    """Fetch transcript via YouTube subtitle API.

    Tries English first, then falls back to any available language.
    """
    from youtube_transcript_api import YouTubeTranscriptApi

    ytt_api = YouTubeTranscriptApi()

    try:
        transcript = ytt_api.fetch(video_id, languages=["en"])
    except Exception:
        # English not available — try any available transcript
        transcript_list = ytt_api.list(video_id)
        available = list(transcript_list)
        if not available:
            raise
        lang = available[0].language_code
        transcript = ytt_api.fetch(video_id, languages=[lang])

    segments = [
        {"text": entry.text, "start": entry.start, "duration": entry.duration}
        for entry in transcript
    ]
    full_text = " ".join(entry.text for entry in transcript)

    return {
        "video_id": video_id,
        "segments": segments,
        "full_text": full_text,
    }


def _fetch_transcripts_api(config: Config, video_ids: list[str], out_dir: Path) -> list[str]:
    """Fetch transcripts using YouTube subtitle API."""
    results = []
    total = len(video_ids)

    print(f"[collect] Fetching transcripts via API for {total} videos...")
    for i, vid in enumerate(video_ids, 1):
        out_file = out_dir / f"{vid}.json"
        if out_file.exists():
            print(f"[collect]   ({i}/{total}) {vid} — cached")
            results.append(vid)
            continue

        try:
            transcript = _fetch_transcript_api(vid)
            out_file.write_text(json.dumps(transcript, indent=2, ensure_ascii=False))
            print(f"[collect]   ({i}/{total}) {vid} — ok")
            results.append(vid)
        except Exception as e:
            print(f"[collect]   ({i}/{total}) {vid} — failed: {e}")

    return results


# -- Local strategy (yt-dlp + whisper) ----------------------------------------


def _fetch_transcripts_local(config: Config, video_ids: list[str], out_dir: Path) -> list[str]:
    """Download audio and transcribe locally with Whisper."""
    from conf_briefing.collect.video_dl import download_audio_batch
    from conf_briefing.collect.whisper_transcribe import transcribe_batch

    audio_dir = config.data_dir / "audio"
    model_name = config.conference.recordings.whisper_model

    # Skip videos that already have transcripts
    to_process = []
    results = []
    for vid in video_ids:
        if (out_dir / f"{vid}.json").exists():
            results.append(vid)
        else:
            to_process.append(vid)

    if not to_process:
        print(f"[collect] All {len(results)} transcripts cached.")
        return results

    cached = len(results)
    if cached:
        print(f"[collect] {cached} transcripts cached, {len(to_process)} to process.")

    # Step 1: Download audio
    print(f"[collect] Downloading audio for {len(to_process)} videos...")
    downloads = download_audio_batch(to_process, audio_dir)

    # Filter to successful downloads
    audio_files = [(vid, path) for vid, path in downloads if path is not None]
    if not audio_files:
        print("[collect] No audio files downloaded, skipping transcription.")
        return results

    # Step 2: Transcribe with Whisper
    print(f"[collect] Transcribing {len(audio_files)} files with Whisper ({model_name})...")
    transcriptions = transcribe_batch(audio_files, model_name=model_name)

    for vid, transcript in transcriptions:
        if transcript is None:
            continue
        transcript["video_id"] = vid
        out_file = out_dir / f"{vid}.json"
        out_file.write_text(json.dumps(transcript, indent=2, ensure_ascii=False))
        results.append(vid)

    return results


# -- Entry point --------------------------------------------------------------


def fetch_transcripts(config: Config) -> Path:
    """Fetch transcripts for all configured videos.

    Uses config.conference.recordings.strategy to choose method:
      - "api": YouTube subtitle API (default, fast, may fail on some videos)
      - "local": Download audio with yt-dlp + transcribe with Whisper
    """
    out_dir = config.data_dir / "transcripts"
    out_dir.mkdir(parents=True, exist_ok=True)

    video_ids = _collect_video_ids(config)
    if not video_ids:
        print("[collect] No video IDs configured, skipping transcripts.")
        return out_dir

    strategy = config.conference.recordings.strategy

    if strategy == "local":
        results = _fetch_transcripts_local(config, video_ids, out_dir)
    elif strategy == "api":
        results = _fetch_transcripts_api(config, video_ids, out_dir)
    else:
        raise ValueError(f"Unknown recordings strategy: {strategy!r} (expected 'api' or 'local')")

    # Write index of all fetched transcripts
    index_path = config.data_dir / "transcripts.json"
    index_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[collect] Saved {len(results)} transcripts to {out_dir}")
    return out_dir
