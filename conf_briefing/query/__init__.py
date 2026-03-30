"""RAG query module: index conference data and ask questions."""

from conf_briefing.config import Config
from conf_briefing.query.ask import ask_question
from conf_briefing.query.chunker import load_chunks
from conf_briefing.query.index import build_index


def run_index(config: Config) -> None:
    """Build the vector index from all analysis data."""
    print(f"[index] Loading chunks from {config.data_dir}...")
    chunks = load_chunks(config)

    if not chunks:
        print("[index] No data found. Run the pipeline first (collect → clean → analyze).")
        return

    # Print chunk type breakdown
    type_counts: dict[str, int] = {}
    for c in chunks:
        type_counts[c.chunk_type] = type_counts.get(c.chunk_type, 0) + 1

    print(f"[index] Loaded {len(chunks)} chunks:")
    for ctype, count in sorted(type_counts.items()):
        print(f"[index]   {ctype}: {count}")

    print(f"[index] Building index at {config.data_dir}/chroma...")
    total = build_index(config, chunks)
    print(f"[index] Done. Indexed {total} chunks.")


def run_ask(
    config: Config,
    question: str,
    top_k: int | None = None,
    chunk_types: list[str] | None = None,
    track: str | None = None,
    verbose: bool = False,
) -> None:
    """Ask a question against the indexed conference data."""
    print(f"[ask] Question: {question}")
    answer = ask_question(
        config,
        question=question,
        top_k=top_k,
        chunk_types=chunk_types,
        track=track,
        verbose=verbose,
    )
    print(f"\n{answer}")
