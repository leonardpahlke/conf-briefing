"""Render Jinja2 templates into markdown reports."""

import json
from pathlib import Path

import jinja2

from conf_briefing.config import Config


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
        print("[report] No templates/ directory found, skipping.")
        return []

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(templates_dir)),
        undefined=jinja2.StrictUndefined,
        keep_trailing_newline=True,
    )

    # Load all analysis data
    context = {
        "conference_name": config.conference.name,
        "schedule": _load_json(data_dir / "schedule_clean.json") or [],
        "agenda": _load_json(data_dir / "analysis_agenda.json") or {},
        "ranking": _load_json(data_dir / "analysis_ranking.json") or [],
        "recordings": _load_json(data_dir / "analysis_recordings.json") or {},
        "talks": _load_json(data_dir / "analysis_talks.json") or [],
    }

    rendered = []
    template_map = {
        "agenda_report.md.j2": "agenda_report.md",
        "ranking_report.md.j2": "ranking_report.md",
        "recording_report.md.j2": "recording_report.md",
    }

    for template_name, output_name in template_map.items():
        try:
            template = env.get_template(template_name)
        except jinja2.TemplateNotFound:
            continue

        out_path = output_dir / output_name
        content = template.render(**context)
        out_path.write_text(content)
        print(f"[report] Rendered {out_path}")
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
    report_section = "\n# Reports\n\n"
    needs_update = False

    for report in reports:
        name = report.stem.replace("_", " ").title()
        link = f"- [{name}](./{report.name})"
        if report.name not in content:
            report_section += link + "\n"
            needs_update = True

    if needs_update:
        if "# Reports" not in content:
            content += report_section
        summary_path.write_text(content)
        print(f"[report] Updated {summary_path}")
