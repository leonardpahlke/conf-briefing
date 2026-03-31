"""YouTube video provider — playlist extraction + yt-dlp download.

Provider interface:
    collect_video_ids(source_url) -> list[str]
    download_videos(video_ids, output_dir) -> list[tuple[str, Path | None]]
"""

import json
import re
from pathlib import Path

from conf_briefing.console import console, progress_bar, tag


def _validate_video_id(vid: str) -> str:
    """Validate a YouTube video ID."""
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,20}", vid):
        raise ValueError(f"Invalid video ID: {vid!r}")
    return vid


def collect_video_ids(source_url: str) -> list[str]:
    """Extract video IDs from a YouTube playlist URL.

    Uses requests to fetch the playlist page and extract video IDs from the HTML.
    """
    import requests

    resp = requests.get(source_url, timeout=30)
    resp.raise_for_status()
    ids = re.findall(r'"videoId":"([a-zA-Z0-9_-]{11})"', resp.text)
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for vid in ids:
        if vid not in seen:
            seen.add(vid)
            unique.append(vid)
    return unique


def _write_metadata(video_id: str, info: dict, output_dir: Path) -> None:
    """Write a {video_id}.meta.json sidecar with video title and metadata."""
    meta = {
        "video_id": video_id,
        "title": info.get("title", ""),
        "uploader": info.get("uploader", ""),
        "upload_date": info.get("upload_date", ""),
        "duration": info.get("duration"),
    }
    meta_path = output_dir / f"{video_id}.meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))


def _ensure_metadata(video_id: str, output_dir: Path) -> None:
    """Backfill metadata for a cached video that has no sidecar yet."""
    meta_path = output_dir / f"{video_id}.meta.json"
    if meta_path.exists():
        return

    import yt_dlp

    opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
            if info:
                _write_metadata(video_id, info, output_dir)
    except Exception:
        # Non-fatal: metadata is best-effort for cached videos
        pass


def _download_single(video_id: str, output_dir: Path) -> Path:
    """Download a single video as 720p MP4.

    Returns the path to the downloaded video file.
    Skips download if the file already exists (cached).
    """
    import yt_dlp

    output_dir.mkdir(parents=True, exist_ok=True)

    # Check cache
    existing = list(output_dir.glob(f"{video_id}.mp4")) or list(output_dir.glob(f"{video_id}.mkv"))
    if existing:
        _ensure_metadata(video_id, output_dir)
        return existing[0]

    outtmpl = str(output_dir / f"{video_id}.%(ext)s")

    opts = {
        "format": "best[height<=720][ext=mp4]/best[height<=720]/best",
        "outtmpl": outtmpl,
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
        if info:
            _write_metadata(video_id, info, output_dir)

    results = list(output_dir.glob(f"{video_id}.mp4")) or list(output_dir.glob(f"{video_id}.*"))
    if not results:
        raise RuntimeError(f"Download succeeded but no video file found for {video_id}")
    return results[0]


def download_videos(
    video_ids: list[str],
    output_dir: Path,
) -> list[tuple[str, Path | None]]:
    """Download videos for multiple video IDs.

    Returns list of (video_id, video_path | None) tuples.
    None means the download failed.
    """
    output_dir = Path(output_dir)
    results: list[tuple[str, Path | None]] = []

    with progress_bar() as pb:
        task = pb.add_task(f"{tag('download')} Downloading videos", total=len(video_ids))
        for vid in video_ids:
            try:
                path = _download_single(vid, output_dir)
                results.append((vid, path))
                pb.update(task, advance=1, description=f"{tag('download')} {vid}")
            except Exception as e:
                results.append((vid, None))
                pb.update(task, advance=1, description=f"{tag('download')} {vid} [red]failed[/red]")
                console.print(f"  {tag('download')} [red]{vid} — {e}[/red]")

    return results
