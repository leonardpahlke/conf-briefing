"""Report generation: Jinja2 templates to markdown."""

from conf_briefing.config import Config
from conf_briefing.report.render import render_reports

__all__ = ["render_reports", "run_report"]


def run_report(config: Config) -> None:
    """Run the full report pipeline."""
    print(f"[report] Generating reports for: {config.conference.name}")
    render_reports(config)
    print("[report] Done.")
