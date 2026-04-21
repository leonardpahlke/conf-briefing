"""Report generation: LLM-driven 4-phase pipeline + legacy templates."""

from conf_briefing.config import Config
from conf_briefing.console import console, tag
from conf_briefing.report.assemble import assemble_report
from conf_briefing.report.draft import draft_sections
from conf_briefing.report.enrich import enrich_sections
from conf_briefing.report.outline import generate_outline
from conf_briefing.report.render import render_reports
from conf_briefing.report.validate import run_validation

__all__ = [
    "generate_outline",
    "draft_sections",
    "enrich_sections",
    "assemble_report",
    "render_reports",
    "run_validation",
    "run_report",
]


def run_report(config: Config) -> None:
    """Run the full LLM-driven report pipeline.

    Phase 1: Generate outline (ranking + global synthesis → structure)
    Phase 2: Draft sections (parallel LLM calls per section)
    Phase 3: Enrich with evidence (RAG transcript quotes for deep dives)
    Phase 4: Assemble (exec summary + Jinja2 final document)

    Legacy templates are also rendered for backward compatibility.
    """
    from conf_briefing.analyze.llm import is_llm_available

    console.rule("[bold yellow]Report[/bold yellow]")
    console.print(f"{tag('report')} Generating reports for: {config.conference.name}")

    if not is_llm_available(config):
        console.print(
            f"{tag('report')} [yellow]LLM not available — falling back to "
            f"legacy template rendering.[/yellow]"
        )
        render_reports(config)
        console.print(f"{tag('report')} Done (legacy only).")
        return

    # Phase 1: Outline
    outline = generate_outline(config)
    if not outline or not outline.get("sections"):
        console.print(
            f"{tag('report')} [yellow]Outline generation failed or empty — "
            f"falling back to legacy templates.[/yellow]"
        )
        render_reports(config)
        console.print(f"{tag('report')} Done (legacy only).")
        return

    # Phase 2: Draft sections
    sections = draft_sections(config, outline)
    if not sections:
        console.print(
            f"{tag('report')} [yellow]No sections drafted — "
            f"falling back to legacy templates.[/yellow]"
        )
        render_reports(config)
        console.print(f"{tag('report')} Done (legacy only).")
        return

    # Phase 3: Enrich with evidence
    enriched = enrich_sections(config, sections, outline)

    # Phase 4: Assemble final report
    assemble_report(config, outline, enriched)

    # Phase 5: Validate
    run_validation(config, enriched)

    # Also render legacy templates
    render_reports(config)

    console.print(f"{tag('report')} Done.")
