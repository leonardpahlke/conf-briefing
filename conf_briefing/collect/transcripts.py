"""Fetch YouTube transcripts for conference recordings."""

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


def fetch_single_transcript(video_id: str) -> dict:
    """Fetch transcript for a single video.

    Tries English first, then falls back to any available language.
    """
    from youtube_transcript_api import YouTubeTranscriptApi

    ytt_api = YouTubeTranscriptApi()

    try:
        transcript = ytt_api.fetch(video_id, languages=["en"])
    except Exception:
        # English not available — try to find any transcript
        transcript_list = ytt_api.list(video_id)
        available = list(transcript_list)
        if not available:
            raise
        # Pick the first available transcript
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


def fetch_transcripts(config: Config) -> Path:
    """Fetch transcripts for all configured videos."""
    data_dir = Path(config.data_dir) / "transcripts"
    data_dir.mkdir(parents=True, exist_ok=True)

    video_ids = list(config.conference.recordings.video_ids)

    if config.conference.recordings.youtube_playlist:
        print("[collect] Extracting video IDs from playlist...")
        playlist_ids = extract_video_ids_from_playlist(
            config.conference.recordings.youtube_playlist
        )
        print(f"[collect] Found {len(playlist_ids)} videos in playlist.")
        video_ids.extend(playlist_ids)

    if not video_ids:
        print("[collect] No video IDs configured, skipping transcripts.")
        return data_dir

    print(f"[collect] Fetching transcripts for {len(video_ids)} videos...")
    results = []
    for i, vid in enumerate(video_ids, 1):
        out_file = data_dir / f"{vid}.json"
        if out_file.exists():
            print(f"[collect]   ({i}/{len(video_ids)}) {vid} — cached")
            results.append(vid)
            continue

        try:
            transcript = fetch_single_transcript(vid)
            out_file.write_text(json.dumps(transcript, indent=2, ensure_ascii=False))
            print(f"[collect]   ({i}/{len(video_ids)}) {vid} — ok")
            results.append(vid)
        except Exception as e:
            print(f"[collect]   ({i}/{len(video_ids)}) {vid} — failed: {e}")

    # Write index of all fetched transcripts
    index_path = Path(config.data_dir) / "transcripts.json"
    index_path.write_text(json.dumps(results, indent=2))
    print(f"[collect] Saved {len(results)} transcripts to {data_dir}")
    return data_dir
