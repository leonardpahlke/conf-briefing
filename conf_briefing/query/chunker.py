"""Load analysis data and split into chunks with metadata for vector indexing."""

import json
from dataclasses import dataclass, field

from conf_briefing.config import Config

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
    return text.lower().replace(" ", "-")[:80]


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

        data = json.loads(transcript_file.read_text())
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


def load_chunks(config: Config) -> list[Chunk]:
    """Load all data sources and produce chunks for indexing."""
    data_dir = config.data_dir
    chunks: list[Chunk] = []

    # Load sessions (matched or clean schedule)
    sessions: list[dict] = []
    for fname in ("matched.json", "schedule_clean.json"):
        p = data_dir / fname
        if p.exists():
            sessions = json.loads(p.read_text())
            break

    # Transcript chunks
    chunks.extend(_load_transcript_chunks(config, sessions))

    # Talk abstract chunks
    chunks.extend(_load_schedule_chunks(sessions))

    # Talk analysis chunks
    talks_path = data_dir / "analysis_talks.json"
    if talks_path.exists():
        analyses = json.loads(talks_path.read_text())
        chunks.extend(_load_talk_analysis_chunks(analyses))

    # Cluster summary chunks
    ranking_path = data_dir / "analysis_ranking.json"
    if ranking_path.exists():
        rankings = json.loads(ranking_path.read_text())
        chunks.extend(_load_cluster_chunks(rankings))

    # Conference narrative chunks
    agenda: dict = {}
    agenda_path = data_dir / "analysis_agenda.json"
    if agenda_path.exists():
        agenda = json.loads(agenda_path.read_text())

    recordings: dict = {}
    recordings_path = data_dir / "analysis_recordings.json"
    if recordings_path.exists():
        recordings = json.loads(recordings_path.read_text())

    chunks.extend(_load_narrative_chunks(agenda, recordings))

    # Ensure every chunk has chunk_type in metadata for ChromaDB filtering
    for c in chunks:
        c.metadata["chunk_type"] = c.chunk_type

    return chunks
