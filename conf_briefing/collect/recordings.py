"""Fetch recordings (video downloads) for conference talks.

Resolves the video provider from the source URL domain and dispatches
to the appropriate provider module. Mirrors the schedule provider
pattern from schedule.py.
"""

import importlib
import json
from pathlib import Path

from conf_briefing.collect.providers import resolve_provider
from conf_briefing.config import Config
from conf_briefing.console import console, tag

# Map URL domain patterns to provider modules.
# Each provider must expose:
#   collect_video_ids(source_url: str) -> list[str]
#   download_videos(video_ids: list[str], output_dir: Path) -> list[tuple[str, Path | None]]
VIDEO_PROVIDER_PATTERNS: list[tuple[str, str]] = [
    ("youtube.com", "conf_briefing.collect.youtube"),
    ("youtu.be", "conf_briefing.collect.youtube"),
]


def _resolve_provider(url: str) -> tuple[str, str] | None:
    """Match a URL to a video provider."""
    return resolve_provider(url, VIDEO_PROVIDER_PATTERNS)


def fetch_recordings(config: Config) -> Path:
    """Download videos for all configured recordings.

    Downloads 720p MP4 videos to {data_dir}/videos/.
    Writes an index file recordings.json listing downloaded video IDs.
    """
    video_dir = config.data_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    rec = config.conference.recordings
    video_ids: list[str] = []

    # Load the provider module if we have a source URL
    provider = None
    if rec.source_url:
        match = _resolve_provider(rec.source_url)
        if match is None:
            raise ValueError(
                f"No video provider found for URL: {rec.source_url}\n"
                f"Supported: {', '.join(d for d, _ in VIDEO_PROVIDER_PATTERNS)}"
            )
        provider_name, module_path = match
        provider = importlib.import_module(module_path)
        console.print(f"{tag('collect')} Using {provider_name} provider for recordings")

        console.print(f"{tag('collect')} Extracting video IDs from source URL...")
        playlist_ids = provider.collect_video_ids(rec.source_url)
        console.print(f"{tag('collect')} Found {len(playlist_ids)} videos.")
        video_ids.extend(playlist_ids)

    # Also include any directly configured video IDs
    video_ids.extend(rec.video_ids)

    if not video_ids:
        console.print(f"{tag('collect')} No video IDs configured, skipping recordings.")
        return video_dir

    # Use the resolved provider for downloading, or fall back to resolving from the first ID
    if provider is None:
        # Direct video IDs without source_url — default to YouTube
        provider = importlib.import_module("conf_briefing.collect.youtube")

    console.print(f"{tag('collect')} Downloading {len(video_ids)} videos...")
    downloads = provider.download_videos(video_ids, video_dir)

    successful = [vid for vid, path in downloads if path is not None]
    failed = len(downloads) - len(successful)

    # Write index
    index_path = config.data_dir / "recordings.json"
    index_path.write_text(json.dumps(successful, indent=2, ensure_ascii=False))

    console.print(
        f"{tag('collect')} Downloaded {len(successful)} videos"
        + (f" ({failed} failed)" if failed else "")
        + f" → {video_dir}"
    )
    return video_dir
