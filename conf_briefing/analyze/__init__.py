"""LLM-based analysis: per-talk, clustering, ranking, synthesis."""

from conf_briefing.analyze.agenda import cluster_talks
from conf_briefing.analyze.ranking import rank_clusters
from conf_briefing.analyze.recordings import analyze_talks
from conf_briefing.analyze.synthesis import synthesize_clusters, synthesize_global
from conf_briefing.config import Config
from conf_briefing.console import console, tag

__all__ = [
    "analyze_talks",
    "cluster_talks",
    "rank_clusters",
    "synthesize_clusters",
    "synthesize_global",
    "run_analyze",
]


def run_analyze(config: Config) -> None:
    """Run the full analysis pipeline."""
    from conf_briefing.analyze.llm import is_llm_available

    console.rule("[bold magenta]Analyze[/bold magenta]")

    if not is_llm_available(config):
        raise RuntimeError(
            f"Ollama not reachable or model '{config.llm.model}' not found. "
            f"Start Ollama and pull the model, or use --skip-analyze."
        )

    console.print(f"{tag('analyze')} Analyzing data for: {config.conference.name}")
    analyze_talks(config)          # Phase 1: per-talk analysis
    cluster_talks(config)          # Phase 2: content-based clustering
    rank_clusters(config)          # Phase 3: cluster ranking
    synthesize_clusters(config)    # Phase 4: per-cluster synthesis
    synthesize_global(config)      # Phase 5: global synthesis
    console.print(f"{tag('analyze')} Done.")
