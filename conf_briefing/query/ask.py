"""RAG question answering: retrieve chunks and generate answers via Claude."""

from conf_briefing.analyze.llm import query_llm
from conf_briefing.config import Config
from conf_briefing.query.index import query_index

SYSTEM_PROMPT = """\
You are a conference analysis assistant. Answer the user's question based on \
the provided context from conference data (talk transcripts, analyses, \
schedules, and cluster summaries).

Rules:
- Only use information present in the provided context.
- If the context doesn't contain enough information, say so clearly.
- Cite specific talk titles when referencing information.
- For transcript-based answers, include video timestamps when available.
- Be concise and direct."""

CONTEXT_TEMPLATE = """\
Context from conference data:

{context}

---

Question: {question}

Answer the question based on the context above. End with a "Sources" section \
listing the talk titles and source types you referenced."""


def _format_chunk(hit: dict, idx: int) -> str:
    """Format a single retrieved chunk for the context window."""
    meta = hit["metadata"]
    header_parts = [f"[{idx + 1}]"]
    header_parts.append(f"type={hit['id'].split(':')[0]}")

    title = meta.get("talk_title", "")
    if title:
        header_parts.append(f"talk={title}")

    track = meta.get("track", "")
    if track:
        header_parts.append(f"track={track}")

    start_time = meta.get("start_time")
    video_id = meta.get("video_id")
    if start_time and video_id:
        minutes = int(float(start_time)) // 60
        seconds = int(float(start_time)) % 60
        header_parts.append(f"time={minutes}:{seconds:02d}")

    header = " | ".join(header_parts)
    return f"{header}\n{hit['text']}"


def ask_question(
    config: Config,
    question: str,
    top_k: int | None = None,
    chunk_types: list[str] | None = None,
    track: str | None = None,
    verbose: bool = False,
) -> str:
    """Retrieve relevant chunks and generate an answer.

    Returns the formatted answer string.
    """
    # Retrieve
    hits = query_index(
        config,
        query=question,
        top_k=top_k,
        chunk_types=chunk_types,
        track=track,
    )

    if not hits:
        return "No relevant information found in the index. Try running `index` first."

    if verbose:
        print(f"\n[ask] Retrieved {len(hits)} chunks:")
        for i, hit in enumerate(hits):
            meta = hit["metadata"]
            print(
                f"  [{i + 1}] {hit['id']}  "
                f"(distance={hit['distance']:.3f}, "
                f"title={meta.get('talk_title', 'N/A')!r})"
            )
        print()

    # Build context
    context_parts = [_format_chunk(hit, i) for i, hit in enumerate(hits)]
    context = "\n\n---\n\n".join(context_parts)

    prompt = CONTEXT_TEMPLATE.format(context=context, question=question)

    # Generate answer
    answer = query_llm(config, SYSTEM_PROMPT, prompt, max_tokens=4096)
    return answer
