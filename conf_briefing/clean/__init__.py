"""Data cleaning and normalization."""

from conf_briefing.clean.normalize import match_transcripts, normalize_schedule
from conf_briefing.config import Config
from conf_briefing.console import console, tag

__all__ = ["match_transcripts", "normalize_schedule", "run_clean"]


def run_clean(config: Config) -> None:
    """Run the full cleaning pipeline."""
    console.rule("[bold green]Clean[/bold green]")
    console.print(f"{tag('clean')} Cleaning data for: {config.conference.name}")
    normalize_schedule(config)
    match_transcripts(config)
    console.print(f"{tag('clean')} Done.")
