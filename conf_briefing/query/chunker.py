"""Load analysis data and split into chunks with metadata for vector indexing."""

import hashlib
import json
from dataclasses import dataclass, field

from conf_briefing.config import Config
from conf_briefing.io import load_json_file

# Approximate chars per token for chunk size estimation
CHARS_PER_TOKEN = 4
CHUNK_TOKENS = 500
OVERLAP_TOKENS = 100
CHUNK_CHARS = CHUNK_TOKENS * CHARS_PER_TOKEN
OVERLAP_CHARS = OVERLAP_TOKENS * CHARS_PER_TOKEN


@dataclass
class Chunk:
    id: str
    text: str
    chunk_type: str
    metadata: dict = field(default_factory=dict)


def _speakers_str(speakers: list[dict]) -> str:
    """Format speakers list into a string."""
    parts = []
    for s in speakers:
        name = s.get("name", "")
        company = s.get("company", "")
        parts.append(f"{name} ({company})" if company else name)
    return ", ".join(parts) if parts else ""


def _companies_str(speakers: list[dict]) -> str:
    """Extract unique companies from speakers."""
    companies = {s.get("company", "") for s in speakers}
    return ", ".join(sorted(c for c in companies if c))


def _base_metadata(session: dict) -> dict:
    """Build common metadata from a session/talk entry."""
    return {
        "talk_title": session.get("title", ""),
        "speakers": _speakers_str(session.get("speakers", [])),
        "companies": _companies_str(session.get("speakers", [])),
        "track": session.get("track", ""),
    }


def _safe_id(text: str) -> str:
    """Make a string safe for use as a chunk ID component."""
    slug = text.lower().replace(" ", "-")[:80]
    suffix = hashlib.md5(text.encode()).hexdigest()[:8]
    return f"{slug}-{suffix}"


def _chunk_transcript_segments(segments: list[dict], video_id: str, session: dict) -> list[Chunk]:
    """Split transcript segments into overlapping chunks."""
    if not segments:
        return []

    chunks = []
    meta = _base_metadata(session)
    meta["video_id"] = video_id
    meta["source_file"] = f"transcripts/{video_id}.json"

    # Build a list of (start_time, text) pairs
    parts: list[tuple[float, str]] = []
    for seg in segments:
        parts.append((seg.get("start", 0.0), seg.get("text", "")))

    # Sliding window over joined text, tracking start times
    buf_text = ""
    buf_start = 0.0
    chunk_idx = 0
    seg_idx = 0

    while seg_idx < len(parts):
        # Fill buffer up to CHUNK_CHARS
        while seg_idx < len(parts) and len(buf_text) < CHUNK_CHARS:
            if not buf_text:
                buf_start = parts[seg_idx][0]
            buf_text += (" " if buf_text else "") + parts[seg_idx][1]
            seg_idx += 1

        chunk_meta = {**meta, "chunk_index": chunk_idx, "start_time": buf_start}
        chunks.append(
            Chunk(
                id=f"transcript_segment:{video_id}:{chunk_idx}",
                text=buf_text.strip(),
                chunk_type="transcript_segment",
                metadata=chunk_meta,
            )
        )
        chunk_idx += 1

        # Slide window: keep overlap portion
        if len(buf_text) > OVERLAP_CHARS:
            buf_text = buf_text[-OVERLAP_CHARS:]
            buf_start = parts[max(0, seg_idx - 1)][0]
        else:
            buf_text = ""

    return chunks


def _load_transcript_chunks(config: Config, sessions: list[dict]) -> list[Chunk]:
    """Load transcript segments and chunk them."""
    data_dir = config.data_dir
    transcripts_dir = data_dir / "transcripts"
    chunks = []

    for session in sessions:
        video_id = session.get("video_id")
        if not video_id:
            continue

        transcript_file = transcripts_dir / f"{video_id}.json"
        if not transcript_file.exists():
            continue

        data = load_json_file(transcript_file)
        segments = data.get("segments", [])
        chunks.extend(_chunk_transcript_segments(segments, video_id, session))

    return chunks


def _load_schedule_chunks(sessions: list[dict]) -> list[Chunk]:
    """Create one chunk per talk abstract."""
    chunks = []
    for session in sessions:
        abstract = session.get("abstract", "").strip()
        if not abstract:
            continue
        title = session.get("title", "")
        meta = _base_metadata(session)
        meta["source_file"] = "schedule_clean.json"
        text = f"{title}\n\n{abstract}"
        chunks.append(
            Chunk(
                id=f"talk_abstract:{_safe_id(title)}:0",
                text=text,
                chunk_type="talk_abstract",
                metadata=meta,
            )
        )
    return chunks


def _load_talk_analysis_chunks(analyses: list[dict]) -> list[Chunk]:
    """Create chunks from individual talk analyses."""
    chunks = []
    for talk in analyses:
        title = talk.get("title", "")
        sid = _safe_id(title)

        base_meta = {"talk_title": title, "source_file": "analysis_talks.json"}

        # Summary chunk
        summary = talk.get("summary", "")
        if summary:
            chunks.append(
                Chunk(
                    id=f"talk_summary:{sid}:0",
                    text=f"{title}\n\n{summary}",
                    chunk_type="talk_summary",
                    metadata={**base_meta},
                )
            )

        # Takeaways chunk
        takeaways = talk.get("key_takeaways", [])
        if takeaways:
            text = f"{title} — Key Takeaways\n\n" + "\n".join(f"- {t}" for t in takeaways)
            chunks.append(
                Chunk(
                    id=f"talk_takeaways:{sid}:0",
                    text=text,
                    chunk_type="talk_takeaways",
                    metadata={**base_meta},
                )
            )

        # Q&A chunk
        qa = talk.get("qa_highlights", [])
        if qa:
            text = f"{title} — Q&A Highlights\n\n" + "\n".join(f"- {q}" for q in qa)
            chunks.append(
                Chunk(
                    id=f"talk_qa:{sid}:0",
                    text=text,
                    chunk_type="talk_qa",
                    metadata={**base_meta},
                )
            )

        # Signals chunk (problems + tools + signals combined)
        problems = talk.get("problems_discussed", [])
        tools = talk.get("tools_and_projects", [])
        signals = talk.get("emerging_signals", [])
        if problems or tools or signals:
            parts = [f"{title} — Signals & Tools"]
            if problems:
                parts.append("Problems: " + "; ".join(problems))
            if tools:
                parts.append("Tools & Projects: " + "; ".join(tools))
            if signals:
                parts.append("Emerging Signals: " + "; ".join(signals))
            chunks.append(
                Chunk(
                    id=f"talk_signals:{sid}:0",
                    text="\n\n".join(parts),
                    chunk_type="talk_signals",
                    metadata={**base_meta},
                )
            )

    return chunks


def _load_cluster_chunks(rankings: list[dict]) -> list[Chunk]:
    """Create one chunk per ranked cluster."""
    chunks = []
    for cluster in rankings:
        name = cluster.get("name", "")
        summary = cluster.get("summary", "")
        argument = cluster.get("relevance_argument", "")
        recommended = cluster.get("recommended_talks", [])
        if not summary:
            continue

        parts = [f"Cluster: {name}", summary]
        if argument:
            parts.append(f"Relevance: {argument}")
        if recommended:
            parts.append("Recommended talks: " + ", ".join(recommended))

        meta = {
            "talk_title": name,
            "source_file": "analysis_ranking.json",
            "relevance_score": str(cluster.get("relevance_score", "")),
        }
        chunks.append(
            Chunk(
                id=f"cluster_summary:{_safe_id(name)}:0",
                text="\n\n".join(parts),
                chunk_type="cluster_summary",
                metadata=meta,
            )
        )
    return chunks


def _load_narrative_chunks(agenda: dict, recordings: dict) -> list[Chunk]:
    """Create chunks from conference-level narratives and themes."""
    chunks = []

    # Agenda narrative
    narrative = agenda.get("narrative", "")
    if narrative:
        chunks.append(
            Chunk(
                id="conference_narrative:agenda:0",
                text=f"Conference Agenda Narrative\n\n{narrative}",
                chunk_type="conference_narrative",
                metadata={"source_file": "analysis_agenda.json"},
            )
        )

    # Recordings narrative
    narrative = recordings.get("narrative", "")
    if narrative:
        chunks.append(
            Chunk(
                id="conference_narrative:recordings:0",
                text=f"Conference Recordings Narrative\n\n{narrative}",
                chunk_type="conference_narrative",
                metadata={"source_file": "analysis_recordings.json"},
            )
        )

    # Cross-cutting themes
    themes = recordings.get("cross_cutting_themes", [])
    for i, theme in enumerate(themes):
        name = theme.get("theme", "")
        desc = theme.get("description", "")
        talks = theme.get("supporting_talks", [])
        if not desc:
            continue
        parts = [f"Cross-cutting Theme: {name}", desc]
        if talks:
            parts.append("Supporting talks: " + ", ".join(talks))
        chunks.append(
            Chunk(
                id=f"conference_narrative:theme:{i}",
                text="\n\n".join(parts),
                chunk_type="conference_narrative",
                metadata={"source_file": "analysis_recordings.json", "talk_title": name},
            )
        )

    return chunks


def _load_slide_chunks(config: Config, sessions: list[dict]) -> list[Chunk]:
    """Create searchable chunks from slide OCR + VLM descriptions."""
    data_dir = config.data_dir
    slides_dir = data_dir / "slides"
    chunks = []

    if not slides_dir.exists():
        return chunks

    for session in sessions:
        video_id = session.get("video_id")
        if not video_id:
            continue

        slides_file = slides_dir / f"{video_id}.json"
        if not slides_file.exists():
            continue

        data = load_json_file(slides_file)
        slides = data.get("slides", [])
        if not slides:
            continue

        meta = _base_metadata(session)
        meta["video_id"] = video_id
        meta["source_file"] = f"slides/{video_id}.json"

        # Group slides into chunks of 3 for reasonable chunk sizes
        for i in range(0, len(slides), 3):
            group = slides[i : i + 3]
            parts = []
            timestamps = []
            for slide in group:
                text = slide.get("text", "").strip()
                desc = slide.get("description", "").strip()
                if text or desc:
                    slide_parts = []
                    if text:
                        slide_parts.append(text)
                    if desc:
                        slide_parts.append(f"[Visual: {desc}]")
                    parts.append("\n".join(slide_parts))
                    timestamps.append(slide.get("timestamp_sec", 0.0))

            if not parts:
                continue

            chunk_idx = i // 3
            chunk_meta = {
                **meta,
                "chunk_index": chunk_idx,
                "start_time": timestamps[0] if timestamps else 0.0,
            }
            chunks.append(
                Chunk(
                    id=f"slide_content:{video_id}:{chunk_idx}",
                    text="\n\n".join(parts),
                    chunk_type="slide_content",
                    metadata=chunk_meta,
                )
            )

    return chunks


def _load_maturity_chunks(recordings: dict) -> list[Chunk]:
    """Create one chunk per technology in maturity_landscape."""
    chunks = []
    for i, item in enumerate(recordings.get("maturity_landscape", [])):
        tech = item.get("technology", "")
        if not tech:
            continue
        parts = [
            f"Technology Maturity: {tech}",
            f"Ring: {item.get('ring', 'assess')}",
            f"Evidence quality: {item.get('evidence_quality', 'anecdotal')}",
        ]
        rationale = item.get("rationale", "")
        if rationale:
            parts.append(f"Rationale: {rationale}")
        talks = item.get("supporting_talks", [])
        if talks:
            parts.append(f"Supporting talks: {', '.join(talks)}")
        chunks.append(
            Chunk(
                id=f"maturity_assessment:{_safe_id(tech)}:{i}",
                text="\n\n".join(parts),
                chunk_type="maturity_assessment",
                metadata={
                    "source_file": "analysis_recordings.json",
                    "talk_title": tech,
                    "ring": item.get("ring", "assess"),
                },
            )
        )
    return chunks


def _load_tension_chunks(recordings: dict) -> list[Chunk]:
    """Create one chunk per debate in tensions."""
    chunks = []
    for i, tension in enumerate(recordings.get("tensions", [])):
        topic = tension.get("topic", "")
        if not topic:
            continue
        side_a = tension.get("side_a", {})
        side_b = tension.get("side_b", {})
        parts = [
            f"Tension: {topic}",
            f"Side A: {side_a.get('position', '')}",
            f"Supporting talks (A): {', '.join(side_a.get('supporting_talks', []))}",
            f"Side B: {side_b.get('position', '')}",
            f"Supporting talks (B): {', '.join(side_b.get('supporting_talks', []))}",
            f"Severity: {tension.get('severity', 'minor')}",
        ]
        implication = tension.get("implication", "")
        if implication:
            parts.append(f"Implication: {implication}")
        chunks.append(
            Chunk(
                id=f"tension:{_safe_id(topic)}:{i}",
                text="\n\n".join(parts),
                chunk_type="tension",
                metadata={
                    "source_file": "analysis_recordings.json",
                    "talk_title": topic,
                    "severity": tension.get("severity", "minor"),
                },
            )
        )
    return chunks


def _load_action_chunks(recordings: dict) -> list[Chunk]:
    """Create one chunk per recommended action."""
    chunks = []
    for i, action in enumerate(recordings.get("recommended_actions", [])):
        text = action.get("action", "")
        if not text:
            continue
        parts = [
            f"Recommended Action: {text}",
            f"Category: {action.get('category', 'watch')}",
            f"Urgency: {action.get('urgency', 'long_term')}",
        ]
        evidence = action.get("supporting_evidence", "")
        if evidence:
            parts.append(f"Evidence: {evidence}")
        chunks.append(
            Chunk(
                id=f"recommended_action:{_safe_id(text)}:{i}",
                text="\n\n".join(parts),
                chunk_type="recommended_action",
                metadata={
                    "source_file": "analysis_recordings.json",
                    "talk_title": text[:80],
                    "category": action.get("category", "watch"),
                    "urgency": action.get("urgency", "long_term"),
                },
            )
        )
    return chunks


def _load_relationship_chunks(recordings: dict) -> list[Chunk]:
    """Create one chunk per technology relationship."""
    chunks = []
    for i, rel in enumerate(recordings.get("technology_relationships", [])):
        entity_a = rel.get("entity_a", "")
        entity_b = rel.get("entity_b", "")
        if not entity_a or not entity_b:
            continue
        relation = rel.get("relation", "")
        talks = rel.get("supporting_talks", [])
        parts = [
            f"Technology Relationship: {entity_a} {relation} {entity_b}",
        ]
        if talks:
            parts.append(f"Supporting talks: {', '.join(talks)}")
        chunks.append(
            Chunk(
                id=f"technology_relationship:{_safe_id(entity_a)}-{_safe_id(entity_b)}:{i}",
                text="\n\n".join(parts),
                chunk_type="technology_relationship",
                metadata={
                    "source_file": "analysis_recordings.json",
                    "talk_title": f"{entity_a} {relation} {entity_b}",
                    "relation": relation,
                },
            )
        )
    return chunks


def load_chunks(config: Config) -> list[Chunk]:
    """Load all data sources and produce chunks for indexing."""
    data_dir = config.data_dir
    chunks: list[Chunk] = []

    # Load sessions (matched or clean schedule)
    sessions: list[dict] = []
    for fname in ("matched.json", "schedule_clean.json"):
        p = data_dir / fname
        if p.exists():
            sessions = load_json_file(p)
            break

    # Transcript chunks
    chunks.extend(_load_transcript_chunks(config, sessions))

    # Slide content chunks
    chunks.extend(_load_slide_chunks(config, sessions))

    # Talk abstract chunks
    chunks.extend(_load_schedule_chunks(sessions))

    # Talk analysis chunks
    talks_path = data_dir / "analysis_talks.json"
    if talks_path.exists():
        analyses = load_json_file(talks_path)
        chunks.extend(_load_talk_analysis_chunks(analyses))

    # Cluster summary chunks
    ranking_path = data_dir / "analysis_ranking.json"
    if ranking_path.exists():
        rankings = load_json_file(ranking_path)
        chunks.extend(_load_cluster_chunks(rankings))

    # Conference narrative chunks
    agenda: dict = {}
    agenda_path = data_dir / "analysis_agenda.json"
    if agenda_path.exists():
        agenda = load_json_file(agenda_path)

    recordings: dict = {}
    recordings_path = data_dir / "analysis_recordings.json"
    if recordings_path.exists():
        recordings = load_json_file(recordings_path)

    chunks.extend(_load_narrative_chunks(agenda, recordings))

    # Maturity, tension, and action chunks from synthesis
    chunks.extend(_load_maturity_chunks(recordings))
    chunks.extend(_load_tension_chunks(recordings))
    chunks.extend(_load_action_chunks(recordings))
    chunks.extend(_load_relationship_chunks(recordings))

    # Ensure every chunk has chunk_type in metadata for ChromaDB filtering
    for c in chunks:
        c.metadata["chunk_type"] = c.chunk_type

    return chunks
