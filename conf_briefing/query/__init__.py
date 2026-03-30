"""RAG query module: index conference data and ask questions."""

from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from conf_briefing.config import Config
from conf_briefing.console import console, tag
from conf_briefing.query.ask import ask_question
from conf_briefing.query.chunker import load_chunks
from conf_briefing.query.index import build_index


def run_index(config: Config) -> None:
    """Build the vector index from all analysis data."""
    console.rule("[bold bright_cyan]Index[/bold bright_cyan]")
    console.print(f"{tag('index')} Loading chunks from {config.data_dir}...")
    chunks = load_chunks(config)

    if not chunks:
        console.print(
            f"{tag('index')} No data found. Run the pipeline first (collect → clean → analyze)."
        )
        return

    # Chunk type breakdown as a table
    type_counts: dict[str, int] = {}
    for c in chunks:
        type_counts[c.chunk_type] = type_counts.get(c.chunk_type, 0) + 1

    table = Table(title=f"Loaded {len(chunks)} chunks")
    table.add_column("Type", style="cyan")
    table.add_column("Count", justify="right", style="bold")
    for ctype, count in sorted(type_counts.items()):
        table.add_row(ctype, str(count))
    console.print(table)

    console.print(f"{tag('index')} Building index at {config.data_dir}/chroma...")
    total = build_index(config, chunks)
    console.print(f"{tag('index')} Done. Indexed {total} chunks.")


def run_ask(
    config: Config,
    question: str,
    top_k: int | None = None,
    chunk_types: list[str] | None = None,
    track: str | None = None,
    verbose: bool = False,
) -> None:
    """Ask a question against the indexed conference data."""
    console.print(f"{tag('ask')} Question: {question}")
    with console.status(f"{tag('ask')} Thinking..."):
        answer = ask_question(
            config,
            question=question,
            top_k=top_k,
            chunk_types=chunk_types,
            track=track,
            verbose=verbose,
        )
    console.print()
    console.print(Panel(Markdown(answer), title="Answer", border_style="green"))
