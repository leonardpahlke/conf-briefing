"""Render Jinja2 templates into markdown reports."""

import json
from pathlib import Path

import jinja2

from conf_briefing.config import Config
from conf_briefing.console import console, tag


def _load_json(path: Path) -> dict | list | None:
    """Load JSON file if it exists."""
    if path.exists():
        return json.loads(path.read_text())
    return None


def render_reports(config: Config) -> list[Path]:
    """Render all report templates with analysis data."""
    data_dir = config.data_dir
    templates_dir = Path(__file__).resolve().parent.parent.parent / "templates"
    output_dir = data_dir / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not templates_dir.exists():
        console.print(f"{tag('report')} No templates/ directory found, skipping.")
        return []

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(templates_dir)),
        undefined=jinja2.Undefined,
        keep_trailing_newline=True,
    )

    # Load all analysis data
    agenda = _load_json(data_dir / "analysis_agenda.json") or {}
    ranking = _load_json(data_dir / "analysis_ranking.json") or []
    recordings = _load_json(data_dir / "analysis_recordings.json") or {}
    talks = _load_json(data_dir / "analysis_talks.json") or []

    context = {
        "conference_name": config.conference.name,
        "schedule": _load_json(data_dir / "schedule_clean.json") or [],
        "agenda": agenda,
        "ranking": ranking,
        "recordings": recordings,
        "talks": talks,
        "has_agenda": bool(agenda),
        "has_recordings": bool(recordings.get("narrative") or recordings.get("cross_cutting_themes")),
        "has_ranking": bool(ranking),
    }

    rendered = []
    template_map = {
        "briefing_report.md.j2": "briefing_report.md",
        "agenda_report.md.j2": "agenda_report.md",
        "ranking_report.md.j2": "ranking_report.md",
    }

    for template_name, output_name in template_map.items():
        try:
            template = env.get_template(template_name)
        except jinja2.TemplateNotFound:
            continue

        out_path = output_dir / output_name
        content = template.render(**context)
        out_path.write_text(content)
        console.print(f"{tag('report')} Rendered {out_path}")
        rendered.append(out_path)

    # Update SUMMARY.md if reports were generated
    if rendered:
        _update_summary(output_dir, rendered)

    return rendered


def _update_summary(docs_src: Path, reports: list[Path]) -> None:
    """Add generated reports to SUMMARY.md if not already present."""
    summary_path = docs_src / "SUMMARY.md"
    if not summary_path.exists():
        return

    content = summary_path.read_text()
    new_links = ""
    for report in reports:
        if report.name not in content:
            name = report.stem.replace("_", " ").title()
            new_links += f"- [{name}](./{report.name})\n"

    if not new_links:
        return

    if "# Reports" not in content:
        content += f"\n# Reports\n\n{new_links}"
    else:
        content = content.rstrip() + "\n" + new_links

    summary_path.write_text(content)
    console.print(f"{tag('report')} Updated {summary_path}")
