"""Data cleaning and normalization."""

from conf_briefing.clean.normalize import match_transcripts, normalize_schedule
from conf_briefing.config import Config

__all__ = ["match_transcripts", "normalize_schedule", "run_clean"]


def run_clean(config: Config) -> None:
    """Run the full cleaning pipeline."""
    print(f"[clean] Cleaning data for: {config.conference.name}")
    normalize_schedule(config)
    match_transcripts(config)
    print("[clean] Done.")
