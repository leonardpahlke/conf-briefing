"""Download video/audio from YouTube and other platforms via yt-dlp.

This module is intentionally decoupled from the rest of the project.
It only depends on yt-dlp and returns file paths.

Usage:
    from conf_briefing.collect.video_dl import download_audio, download_audio_batch
    path = download_audio("dQw4w9WgXcQ", output_dir="events/myconf/audio")
"""

from pathlib import Path


def download_audio(video_id: str, output_dir: str | Path) -> Path:
    """Download audio-only for a single video.

    Returns the path to the downloaded audio file.
    Skips download if the file already exists (cached).
    """
    import yt_dlp

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check cache — yt-dlp adds the extension, so glob for any match
    existing = list(output_dir.glob(f"{video_id}.*"))
    if existing:
        return existing[0]

    outtmpl = str(output_dir / f"{video_id}.%(ext)s")

    opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

    # Find the output file (yt-dlp may change extension)
    results = list(output_dir.glob(f"{video_id}.*"))
    if not results:
        raise RuntimeError(f"Download succeeded but no file found for {video_id}")
    return results[0]


def download_audio_batch(
    video_ids: list[str],
    output_dir: str | Path,
) -> list[tuple[str, Path | None]]:
    """Download audio for multiple videos.

    Returns list of (video_id, audio_path | None) tuples.
    None means the download failed.
    """
    output_dir = Path(output_dir)
    results = []

    total = len(video_ids)
    for i, vid in enumerate(video_ids, 1):
        cached = list(output_dir.glob(f"{vid}.*"))
        if cached:
            print(f"[download]   ({i}/{total}) {vid} — cached")
            results.append((vid, cached[0]))
            continue

        try:
            path = download_audio(vid, output_dir)
            print(f"[download]   ({i}/{total}) {vid} — ok")
            results.append((vid, path))
        except Exception as e:
            print(f"[download]   ({i}/{total}) {vid} — failed: {e}")
            results.append((vid, None))

    return results
