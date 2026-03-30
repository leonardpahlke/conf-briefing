"""Slide extraction: scene detection, deduplication, and OCR."""

import json
from pathlib import Path

from conf_briefing.config import Config
from conf_briefing.console import console, progress_bar, tag


def _extract_frames(video_path: Path, output_dir: Path, threshold: float) -> list[dict]:
    """Detect scene changes and save one frame per scene transition.

    Returns list of dicts with 'index', 'timestamp_sec', 'image_file'.
    """
    import cv2
    from scenedetect import ContentDetector, SceneManager, open_video

    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=15))
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    if not scene_list:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
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
        img = Image.open(frame["_abs_path"])
        h = imagehash.dhash(img)

        is_dup = any(abs(h - prev) <= max_distance for prev in seen_hashes)
        if not is_dup:
            unique.append(frame)
            seen_hashes.append(h)

    return unique


def _ocr_frames(frames: list[dict]) -> list[dict]:
    """Run OCR on each frame image."""
    import pytesseract
    from PIL import Image

    results = []
    for frame in frames:
        img = Image.open(frame["_abs_path"])
        text = pytesseract.image_to_string(img).strip()
        results.append(
            {
                "index": frame["index"],
                "timestamp_sec": frame["timestamp_sec"],
                "image_file": frame["image_file"],
                "text": text,
            }
        )
    return results


def extract_slides(video_path: Path, output_dir: Path, threshold: float = 27.0) -> Path:
    """Extract slides from a single video. Returns path to slides JSON."""
    video_id = video_path.stem
    frames_dir = output_dir / video_id

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

    # Filter out already-processed videos
    to_process = []
    existing = []
    for vf in video_files:
        slides_path = output_dir / f"{vf.stem}.json"
        if slides_path.exists():
            existing.append(slides_path)
        else:
            to_process.append(vf)

    if existing:
        console.print(f"{tag('slides')} Skipping {len(existing)} already-processed video(s).")

    if not to_process:
        console.print(f"{tag('slides')} All videos already processed.")
        return existing

    threshold = config.extract.scene_threshold
    console.print(
        f"{tag('slides')} Extracting slides from {len(to_process)} video(s) "
        f"(threshold={threshold})."
    )

    results = list(existing)
    with progress_bar() as pb:
        task = pb.add_task(f"{tag('slides')} Extracting slides", total=len(to_process))
        for vf in to_process:
            console.print(f"{tag('slides')} Processing: {vf.name}")
            out = extract_slides(vf, output_dir, threshold)
            results.append(out)
            pb.advance(task)

    console.print(f"{tag('slides')} Extracted slides from {len(to_process)} video(s).")
    return results
