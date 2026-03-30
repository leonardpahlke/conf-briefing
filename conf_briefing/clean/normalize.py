"""Normalize and clean collected data."""

import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

from conf_briefing.config import Config
from conf_briefing.console import console, tag


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

    sessions = json.loads(schedule_path.read_text())
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


def match_transcripts(config: Config) -> Path:
    """Match transcripts to schedule entries by title similarity."""
    data_dir = config.data_dir
    schedule_path = data_dir / "schedule_clean.json"
    transcripts_dir = data_dir / "transcripts"
    out_path = data_dir / "matched.json"

    if not schedule_path.exists():
        console.print(f"{tag('clean')} No cleaned schedule found, skipping transcript matching.")
        return out_path

    sessions = json.loads(schedule_path.read_text())

    if not transcripts_dir.exists():
        console.print(f"{tag('clean')} No transcripts directory found, skipping matching.")
        # Write sessions without transcripts
        out_path.write_text(json.dumps(sessions, indent=2, ensure_ascii=False))
        return out_path

    # Load all transcripts
    transcripts = {}
    for tf in transcripts_dir.glob("*.json"):
        data = json.loads(tf.read_text())
        if "video_id" in data:
            transcripts[data["video_id"]] = data

    console.print(
        f"{tag('clean')} Matching {len(transcripts)} transcripts to {len(sessions)} sessions..."
    )

    # Simple greedy matching: for each transcript, find best matching session
    for session in sessions:
        session["transcript"] = None
        session["video_id"] = None

    matched_count = 0
    for vid, transcript in transcripts.items():
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
            matched_count += 1
        else:
            # Unmatched transcript — add as standalone entry
            sessions.append(
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
                }
            )

    # Match slide data to sessions by video_id
    slides_dir = data_dir / "slides"
    slides_matched = 0
    if slides_dir.exists():
        slides_data = {}
        for sf in slides_dir.glob("*.json"):
            sdata = json.loads(sf.read_text())
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
                if slide_texts or slide_descs:
                    slides_matched += 1

    out_path.write_text(json.dumps(sessions, indent=2, ensure_ascii=False))
    console.print(
        f"{tag('clean')} Matched {matched_count} transcripts, "
        f"{slides_matched} slide sets, saved to {out_path}"
    )
    return out_path
