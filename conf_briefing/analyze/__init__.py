"""LLM-based analysis: agenda, cluster ranking, recordings."""

from conf_briefing.analyze.agenda import analyze_agenda
from conf_briefing.analyze.ranking import rank_clusters
from conf_briefing.analyze.recordings import analyze_recordings
from conf_briefing.config import Config
from conf_briefing.console import console, tag

__all__ = ["analyze_agenda", "analyze_recordings", "rank_clusters", "run_analyze"]


def run_analyze(config: Config) -> None:
    """Run the full analysis pipeline."""
    from conf_briefing.analyze.llm import is_llm_available

    console.rule("[bold magenta]Analyze[/bold magenta]")

    if not is_llm_available(config):
        console.print(
            f"{tag('analyze')} Skipping — Ollama not reachable or model "
            f"'{config.llm.model}' not found. Start Ollama to enable analysis."
        )
        return

    console.print(f"{tag('analyze')} Analyzing data for: {config.conference.name}")
    analyze_agenda(config)
    rank_clusters(config)
    analyze_recordings(config)
    console.print(f"{tag('analyze')} Done.")
