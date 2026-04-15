"""Slide extraction: scene detection, deduplication, and OCR."""

import gc
import json
import time
from pathlib import Path

from conf_briefing.config import MIN_VIDEO_DURATION_SEC, Config
from conf_briefing.console import console, tag
from conf_briefing.io import load_json_file

# Minimum number of frames a scene must last before a new cut is registered
_MIN_SCENE_LEN_FRAMES = 15

# Videos longer than this (seconds) use a lower scene-detection threshold
_KEYNOTE_DURATION_THRESHOLD = 300
_KEYNOTE_THRESHOLD = 20.0


def _extract_frames(video_path: Path, output_dir: Path, threshold: float) -> list[dict]:
    """Detect scene changes and save one frame per scene transition.

    Returns list of dicts with 'index', 'timestamp_sec', 'image_file'.
    """
    import cv2
    from scenedetect import ContentDetector, SceneManager, open_video

    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold, min_scene_len=_MIN_SCENE_LEN_FRAMES)
    )
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    if not scene_list:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        frames = []
        for i, (start, _end) in enumerate(scene_list):
            frame_num = start.get_frames()
            timestamp = frame_num / fps

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                continue

            img_name = f"{i + 1:04d}.jpg"
            img_path = output_dir / img_name
            cv2.imwrite(str(img_path), frame)

            frames.append(
                {
                    "index": i,
                    "timestamp_sec": round(timestamp, 2),
                    "image_file": str(Path("slides") / video_path.stem / img_name),
                    "_abs_path": str(img_path),
                }
            )
    finally:
        cap.release()

    return frames


def _dedup_frames(frames: list[dict], max_distance: int = 6) -> list[dict]:
    """Remove near-duplicate frames using perceptual hashing."""
    import imagehash
    from PIL import Image

    if not frames:
        return []

    unique = []
    seen_hashes: list[imagehash.ImageHash] = []

    for frame in frames:
        with Image.open(frame["_abs_path"]) as img:
            h = imagehash.dhash(img)

        is_dup = any(abs(h - prev) <= max_distance for prev in seen_hashes)
        if not is_dup:
            unique.append(frame)
            seen_hashes.append(h)

    return unique


def _ocr_frames(frames: list[dict]) -> list[dict]:
    """Run Tesseract OCR on each frame image (parallel — Tesseract releases the GIL)."""
    import os
    from concurrent.futures import ThreadPoolExecutor

    import pytesseract
    from PIL import Image

    if not frames:
        return []

    def _ocr_single(idx_frame: tuple[int, dict]) -> tuple[int, dict]:
        idx, frame = idx_frame
        with Image.open(frame["_abs_path"]) as img:
            text = pytesseract.image_to_string(img).strip()
        return idx, {
            "index": frame["index"],
            "timestamp_sec": frame["timestamp_sec"],
            "image_file": frame["image_file"],
            "text": text,
        }

    max_workers = min(len(frames), os.cpu_count() or 4)
    indexed: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for idx, result in pool.map(_ocr_single, enumerate(frames)):
            indexed[idx] = result

    return [indexed[i] for i in range(len(frames))]


def _get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds from .meta.json."""
    meta_path = video_path.with_suffix(".meta.json")
    if meta_path.exists():
        meta = load_json_file(meta_path)
        return float(meta.get("duration", 0))
    return 0.0


def extract_slides(
    video_path: Path,
    output_dir: Path,
    threshold: float = 27.0,
    duration: float = 0.0,
) -> Path:
    """Extract slides from a single video. Returns path to slides JSON."""
    video_id = video_path.stem
    frames_dir = output_dir / video_id

    # Use a lower threshold for longer videos (keynotes) to catch gradual transitions
    if not duration:
        duration = _get_video_duration(video_path)
    if duration > _KEYNOTE_DURATION_THRESHOLD:
        threshold = min(threshold, _KEYNOTE_THRESHOLD)

    # Scene detection → frame extraction
    frames = _extract_frames(video_path, frames_dir, threshold)

    # Deduplication
    unique_frames = _dedup_frames(frames)

    # OCR
    slides = _ocr_frames(unique_frames)

    result = {
        "video_id": video_id,
        "slides": slides,
    }

    out_path = output_dir / f"{video_id}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return out_path


def extract_all_slides(config: Config) -> list[Path]:
    """Extract slides from all downloaded videos. Returns list of slide JSON paths."""
    data_dir = config.data_dir
    videos_dir = data_dir / "videos"
    output_dir = data_dir / "slides"

    if not videos_dir.exists():
        console.print(f"{tag('slides')} No videos directory found, skipping slide extraction.")
        return []

    video_files = sorted(videos_dir.glob("*.mp4"))
    if not video_files:
        console.print(f"{tag('slides')} No video files found.")
        return []

    # Filter out already-processed videos and short videos (highlight reels)
    to_process: list[tuple[Path, float]] = []  # (video_path, duration)
    existing = []
    skipped_short = 0
    for vf in video_files:
        slides_path = output_dir / f"{vf.stem}.json"
        if slides_path.exists():
            existing.append(slides_path)
            continue

        # Check video duration from .meta.json
        duration = _get_video_duration(vf)
        if duration and duration < MIN_VIDEO_DURATION_SEC:
            skipped_short += 1
            continue

        to_process.append((vf, duration))

    if existing:
        console.print(f"{tag('slides')} Skipping {len(existing)} already-processed video(s).")
    if skipped_short:
        console.print(
            f"{tag('slides')} Skipping {skipped_short} short video(s) (<{MIN_VIDEO_DURATION_SEC}s)."
        )

    if not to_process:
        console.print(f"{tag('slides')} All videos already processed.")
        return existing

    threshold = config.extract.scene_threshold
    console.print(
        f"{tag('slides')} Extracting slides from {len(to_process)} video(s) "
        f"(threshold={threshold}, ocr=tesseract)."
    )

    results = list(existing)
    total = len(to_process)
    for i, (vf, duration) in enumerate(to_process, 1):
        with console.status(f"{tag('slides')} [{i}/{total}] Processing {vf.name}..."):
            t0 = time.monotonic()
            out = extract_slides(vf, output_dir, threshold, duration=duration)
            elapsed = time.monotonic() - t0
        console.print(f"{tag('slides')} [{i}/{total}] {vf.name} ({elapsed:.0f}s)")
        results.append(out)
        gc.collect()  # Reclaim OpenCV/PIL memory between videos

    console.print(f"{tag('slides')} Extracted slides from {total} video(s).")
    return results
