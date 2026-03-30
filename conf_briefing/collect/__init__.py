"""Data collection: schedule fetching and transcript downloading."""

from conf_briefing.collect.schedule import fetch_schedule
from conf_briefing.collect.transcripts import fetch_transcripts
from conf_briefing.config import Config

__all__ = ["fetch_schedule", "fetch_transcripts", "run_collect"]


def run_collect(config: Config) -> None:
    """Run the full collection pipeline."""
    print(f"[collect] Collecting data for: {config.conference.name}")
    fetch_schedule(config)
    fetch_transcripts(config)
    print("[collect] Done.")
