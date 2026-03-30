"""LLM-based analysis: agenda, cluster ranking, recordings."""

from conf_briefing.analyze.agenda import analyze_agenda
from conf_briefing.analyze.ranking import rank_clusters
from conf_briefing.analyze.recordings import analyze_recordings
from conf_briefing.config import Config

__all__ = ["analyze_agenda", "analyze_recordings", "rank_clusters", "run_analyze"]


def run_analyze(config: Config) -> None:
    """Run the full analysis pipeline."""
    print(f"[analyze] Analyzing data for: {config.conference.name}")
    analyze_agenda(config)
    rank_clusters(config)
    analyze_recordings(config)
    print("[analyze] Done.")
