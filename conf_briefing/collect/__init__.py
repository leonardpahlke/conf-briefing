"""Data collection: schedule fetching and video downloading."""

from conf_briefing.collect.recordings import fetch_recordings
from conf_briefing.collect.schedule import fetch_schedule
from conf_briefing.config import Config
from conf_briefing.console import console, tag

__all__ = ["fetch_recordings", "fetch_schedule", "run_collect"]


def run_collect(config: Config) -> None:
    """Run the full collection pipeline."""
    console.rule("[bold cyan]Collect[/bold cyan]")
    console.print(f"{tag('collect')} Collecting data for: {config.conference.name}")
    fetch_schedule(config)
    fetch_recordings(config)
    console.print(f"{tag('collect')} Done.")
