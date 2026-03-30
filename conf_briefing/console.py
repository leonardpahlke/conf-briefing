"""Shared Rich console for styled terminal output."""

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

console = Console()
err_console = Console(stderr=True)

_TAG_COLORS = {
    "collect": "cyan",
    "sched": "cyan",
    "download": "cyan",
    "extract": "bright_cyan",
    "whisper": "cyan",
    "slides": "bright_magenta",
    "clean": "green",
    "analyze": "magenta",
    "visualize": "blue",
    "report": "yellow",
    "index": "bright_cyan",
    "ask": "bright_green",
    "preflight": "bright_green",
    "config": "dim",
}


def tag(module: str) -> str:
    """Return a Rich-styled module tag like '[cyan][collect][/]'."""
    color = _TAG_COLORS.get(module, "white")
    return f"[{color}]\\[{module}][/{color}]"


def progress_bar(description: str = "Working...") -> Progress:
    """Create a standard progress bar for batch operations."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    )
