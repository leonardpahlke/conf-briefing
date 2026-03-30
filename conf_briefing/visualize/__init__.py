"""Visualization: charts, diagrams, and word clouds."""

from conf_briefing.config import Config
from conf_briefing.visualize.charts import generate_charts
from conf_briefing.visualize.clouds import generate_wordcloud

__all__ = ["generate_charts", "generate_wordcloud", "run_visualize"]


def run_visualize(config: Config) -> None:
    """Run the full visualization pipeline."""
    print(f"[visualize] Generating visuals for: {config.conference.name}")
    generate_charts(config)
    generate_wordcloud(config)
    print("[visualize] Done.")
