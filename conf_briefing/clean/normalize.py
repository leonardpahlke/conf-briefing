"""Normalize and clean collected data."""

import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

from conf_briefing.config import MIN_VIDEO_DURATION_SEC, Config
from conf_briefing.console import console, tag
from conf_briefing.io import load_json_file


def clean_text(text: str) -> str:
    """Normalize whitespace, encoding, and strip HTML tags."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"<[^>]+>", "", text)  # strip HTML
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_session(session: dict) -> dict:
    """Clean and validate a single session entry."""
    return {
        "title": clean_text(session.get("title", "")),
        "abstract": clean_text(session.get("abstract", "")),
        "speakers": [
            {
                "name": clean_text(s.get("name", "")),
                "company": clean_text(s.get("company", "")),
            }
            for s in session.get("speakers", [])
        ],
        "track": clean_text(session.get("track", "")),
        "format": clean_text(session.get("format", "")),
        "time": session.get("time", ""),
        "tags": session.get("tags", []),
    }


def normalize_schedule(config: Config) -> Path:
    """Normalize the collected schedule data."""
    data_dir = config.data_dir
    schedule_path = data_dir / "schedule.json"

    if not schedule_path.exists():
        console.print(f"{tag('clean')} No schedule.json found, skipping normalization.")
        return schedule_path

    sessions = load_json_file(schedule_path)
    cleaned = [normalize_session(s) for s in sessions]

    # Deduplicate by title
    seen_titles: set[str] = set()
    unique = []
    for s in cleaned:
        key = s["title"].lower()
        if key and key not in seen_titles:
            seen_titles.add(key)
            unique.append(s)

    out_path = data_dir / "schedule_clean.json"
    out_path.write_text(json.dumps(unique, indent=2, ensure_ascii=False))
    console.print(
        f"{tag('clean')} Normalized {len(unique)} sessions (from {len(sessions)}) → {out_path}"
    )
    return out_path


def similarity(a: str, b: str) -> float:
    """Compute string similarity ratio."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _extract_references(slide_texts: list[str]) -> list[str]:
    """Extract URLs and GitHub references from slide OCR text."""
    url_pattern = re.compile(r"https?://[^\s<>\]\)\"']+")
    refs: list[str] = []
    for text in slide_texts:
        refs.extend(url_pattern.findall(text))
    return sorted(set(refs))


def _align_slides_to_transcript(
    transcript_segments: list[dict],
    slide_entries: list[dict],
) -> str:
    """Interleave slides and transcript by timestamp.

    For each slide, finds overlapping transcript segments based on the slide's
    timestamp, producing a chronologically ordered view of slides + speech.
    """
    if not slide_entries or not transcript_segments:
        return ""

    # Build timeline: slides sorted by timestamp
    timed_slides = []
    for i, slide in enumerate(slide_entries):
        ts = slide.get("timestamp_sec", slide.get("timestamp", 0))
        if isinstance(ts, str):
            # Parse "MM:SS" or "HH:MM:SS" format
            parts = ts.split(":")
            try:
                if len(parts) == 2:
                    ts = int(parts[0]) * 60 + int(parts[1])
                elif len(parts) == 3:
                    ts = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                else:
                    ts = 0
            except ValueError:
                ts = 0
        timed_slides.append((float(ts), i, slide))

    timed_slides.sort(key=lambda x: x[0])

    parts: list[str] = []
    seg_idx = 0

    for slide_i, (slide_ts, orig_idx, slide) in enumerate(timed_slides):
        # Determine end time for this slide (= start of next slide, or end of transcript)
        next_ts = (
            timed_slides[slide_i + 1][0] if slide_i + 1 < len(timed_slides) else float("inf")
        )

        # Format slide header
        minutes = int(slide_ts) // 60
        seconds = int(slide_ts) % 60
        header = f"[SLIDE {orig_idx + 1} at {minutes}m{seconds:02d}s]"

        ocr = (slide.get("text") or "").strip()
        desc = (slide.get("description") or "").strip()
        slide_parts = [header]
        if ocr:
            slide_parts.append(f"OCR: {ocr}")
        if desc:
            slide_parts.append(f"VLM: {desc}")

        # Collect transcript segments that overlap with this slide's display time
        transcript_lines: list[str] = []
        while seg_idx < len(transcript_segments):
            seg = transcript_segments[seg_idx]
            seg_start = seg.get("start", 0)
            if seg_start >= next_ts:
                break
            if seg_start >= slide_ts:
                text = seg.get("text", "").strip()
                speaker = seg.get("speaker", "")
                if text:
                    if speaker:
                        transcript_lines.append(f"[{speaker}] {text}")
                    else:
                        transcript_lines.append(text)
            seg_idx += 1

        if transcript_lines:
            slide_parts.append("[TRANSCRIPT]")
            slide_parts.extend(transcript_lines)

        parts.append("\n".join(slide_parts))

    return "\n\n".join(parts)


def match_transcripts(config: Config) -> Path:
    """Match transcripts to schedule entries by title similarity."""
    data_dir = config.data_dir
    schedule_path = data_dir / "schedule_clean.json"
    transcripts_dir = data_dir / "transcripts"
    out_path = data_dir / "matched.json"

    if not schedule_path.exists():
        console.print(f"{tag('clean')} No cleaned schedule found, skipping transcript matching.")
        return out_path

    sessions = load_json_file(schedule_path)

    if not transcripts_dir.exists():
        console.print(f"{tag('clean')} No transcripts directory found, skipping matching.")
        # Write sessions without transcripts
        out_path.write_text(json.dumps(sessions, indent=2, ensure_ascii=False))
        return out_path

    # Load all transcripts and their segments in a single pass
    transcripts = {}
    transcripts_segments: dict[str, list[dict]] = {}
    for tf in transcripts_dir.glob("*.json"):
        data = load_json_file(tf)
        if "video_id" in data:
            transcripts[data["video_id"]] = data
            if "segments" in data:
                transcripts_segments[data["video_id"]] = data["segments"]

    console.print(
        f"{tag('clean')} Matching {len(transcripts)} transcripts to {len(sessions)} sessions..."
    )

    # Simple greedy matching: for each transcript, find best matching session
    for session in sessions:
        session["transcript"] = None
        session["video_id"] = None

    # Filter out short videos (highlight reels, teasers) before matching
    skipped_short = 0
    unmatched: list[dict] = []

    matched_count = 0
    for vid, transcript in transcripts.items():
        duration = transcript.get("duration_sec", 0)
        if duration and duration < MIN_VIDEO_DURATION_SEC:
            skipped_short += 1
            continue

        # Use the first segment or full text to try title matching
        best_score = 0.0
        best_idx = -1
        title = transcript.get("title", "")

        for i, session in enumerate(sessions):
            if session["video_id"] is not None:
                continue
            score = similarity(title, session["title"]) if title else 0.0
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx >= 0 and best_score > 0.6:
            sessions[best_idx]["transcript"] = transcript.get("full_text", "")
            sessions[best_idx]["video_id"] = vid
            sessions[best_idx]["duration_sec"] = duration
            matched_count += 1
        else:
            # Collect unmatched transcripts separately to avoid mutating during iteration
            unmatched.append(
                {
                    "title": title or f"Recording {vid}",
                    "abstract": "",
                    "speakers": [],
                    "track": "",
                    "format": "recording",
                    "time": "",
                    "tags": [],
                    "transcript": transcript.get("full_text", ""),
                    "video_id": vid,
                    "duration_sec": duration,
                }
            )

    sessions.extend(unmatched)

    # Match slide data to sessions by video_id
    slides_dir = data_dir / "slides"
    slides_matched = 0
    if slides_dir.exists():
        slides_data = {}
        for sf in slides_dir.glob("*.json"):
            sdata = load_json_file(sf)
            vid = sdata.get("video_id")
            if vid:
                slides_data[vid] = sdata

        for session in sessions:
            vid = session.get("video_id")
            if vid and vid in slides_data:
                slide_entries = slides_data[vid].get("slides", [])
                slide_texts = [s.get("text", "") for s in slide_entries if s.get("text")]
                slide_descs = [
                    s.get("description", "") for s in slide_entries if s.get("description")
                ]
                session["slide_text"] = "\n\n".join(slide_texts) if slide_texts else ""
                session["slide_descriptions"] = " | ".join(slide_descs) if slide_descs else ""

                # Extract URLs from slide OCR text
                session["slide_references"] = _extract_references(slide_texts)

                # Temporal slide-transcript alignment
                segments = transcripts_segments.get(vid, [])
                if segments and slide_entries:
                    session["slide_aligned"] = _align_slides_to_transcript(
                        segments, slide_entries
                    )

                if slide_texts or slide_descs:
                    slides_matched += 1

    out_path.write_text(json.dumps(sessions, indent=2, ensure_ascii=False))
    console.print(
        f"{tag('clean')} Matched {matched_count} transcripts, "
        f"{slides_matched} slide sets"
        f"{f', skipped {skipped_short} short videos (<{MIN_VIDEO_DURATION_SEC}s)' if skipped_short else ''}"
        f", saved to {out_path}"
    )
    return out_path
