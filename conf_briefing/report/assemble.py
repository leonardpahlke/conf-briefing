"""Phase 4: Executive summary generation + final Jinja2 assembly."""

import json
from datetime import datetime, timezone
from pathlib import Path

import jinja2

from conf_briefing.analyze.llm import query_llm_json
from conf_briefing.config import Config
from conf_briefing.console import console, tag
from conf_briefing.io import load_json_file
from conf_briefing.report.schemas import ExecutiveSummary

EXEC_SYSTEM_PROMPT = """\
You are a technical intelligence analyst writing an executive summary \
for a conference briefing. The audience is a cloud-native engineering team."""

EXEC_SUMMARY_PROMPT = """\
Write an executive summary for the {conference_name} intelligence briefing.

Report thesis: {thesis}

The team's evaluation topics:
{eval_topics}

Here are the section summaries (key_takeaway from each section):
{takeaways}

Here is the full report body for context:
{body_preview}

Produce JSON with:
- summary: A 3-5 paragraph executive summary in markdown. Open with the conference's \
main message. Cover the most important findings across all sections. Close with the \
overall direction and what it means for practitioners. Write for a reader who will \
only read this summary.
- key_findings: 5-7 bullet points, each a single sentence capturing a major finding. \
Be specific — include technology names, company examples, or metrics where possible.
- top_actions: 3-5 most urgent recommended actions for the team. Each should be \
specific and actionable (e.g., "Evaluate Kueue for batch ML scheduling" not \
"Consider new technologies")."""


def _generate_executive_summary(
    config: Config, outline: dict, sections: list[dict],
) -> dict:
    """Generate executive summary via LLM."""
    # Build takeaways from enriched sections (key_takeaway now propagates)
    takeaways = []
    for section in sections:
        kt = section.get("key_takeaway", "")
        if kt:
            takeaways.append(f"- **{section['title']}**: {kt}")

    # Build a body preview (truncated to fit context)
    body_parts = []
    for s in sections:
        body_parts.append(f"## {s['title']}\n\n{s['prose']}")
    body = "\n\n---\n\n".join(body_parts)
    # Truncate to ~15k chars to leave room for prompt overhead
    body_preview = body[:15000]
    if len(body) > 15000:
        body_preview += "\n\n[... truncated for context ...]"

    prompt = EXEC_SUMMARY_PROMPT.format(
        conference_name=config.conference.name,
        thesis=outline.get("thesis", ""),
        eval_topics="\n".join(f"- {t}" for t in config.analyze.eval_topics),
        takeaways="\n".join(takeaways) if takeaways else "No takeaways available.",
        body_preview=body_preview,
    )

    return query_llm_json(
        config, EXEC_SYSTEM_PROMPT, prompt,
        max_tokens=4096, schema=ExecutiveSummary,
    )


def _build_appendix(config: Config, outline: dict) -> list[dict]:
    """Build curated appendix talk list (no LLM — data-driven)."""
    strategy = outline.get("appendix_strategy", config.report.appendix_strategy)
    data_dir = config.data_dir

    talks_path = data_dir / "analysis_talks.json"
    ranking_path = data_dir / "analysis_ranking.json"
    agenda_path = data_dir / "analysis_agenda.json"

    talks = load_json_file(talks_path) if talks_path.exists() else []
    ranking = load_json_file(ranking_path) if ranking_path.exists() else []
    agenda = load_json_file(agenda_path) if agenda_path.exists() else {}

    talks_by_title = {t["title"]: t for t in talks if t.get("title")}
    # Prefix lookup for fuzzy matching
    talks_by_prefix = {}
    for title, talk in talks_by_title.items():
        prefix = title.split(" - ")[0].strip()
        if prefix not in talks_by_prefix:
            talks_by_prefix[prefix] = talk

    def _resolve(title: str) -> dict | None:
        if title in talks_by_title:
            return talks_by_title[title]
        prefix = title.split(" - ")[0].strip()
        if prefix in talks_by_prefix:
            return talks_by_prefix[prefix]
        for full_title, t in talks_by_title.items():
            if full_title.startswith(title):
                return t
        return None

    if strategy == "none":
        return []

    if strategy == "top_talks_only":
        # Include recommended talks from each ranked cluster
        seen = set()
        appendix = []
        for entry in ranking:
            cluster_name = entry.get("name", "")
            for talk_title in entry.get("recommended_talks", []):
                if talk_title in seen:
                    continue
                seen.add(talk_title)
                talk = _resolve(talk_title)
                if talk:
                    appendix.append({
                        "title": talk.get("title", talk_title),
                        "cluster": cluster_name,
                        "summary": talk.get("summary", ""),
                        "key_takeaways": talk.get("key_takeaways", []),
                        "evidence_quality": talk.get("evidence_quality", ""),
                        "speaker_perspective": talk.get("speaker_perspective", ""),
                        "tools_and_projects": talk.get("tools_and_projects", []),
                    })
        return appendix

    if strategy == "by_cluster":
        # Deep-dive clusters get full talk summaries, briefs get titles only
        deep_dive_clusters = {
            s.get("cluster_name")
            for s in outline.get("sections", [])
            if s.get("section_type") == "cluster_deep_dive"
        }

        appendix = []
        for cluster in agenda.get("clusters", []):
            cluster_name = cluster.get("name", "")
            is_deep = cluster_name in deep_dive_clusters
            for talk_title in cluster.get("talks", []):
                talk = _resolve(talk_title)
                if is_deep and talk:
                    appendix.append({
                        "title": talk.get("title", talk_title),
                        "cluster": cluster_name,
                        "summary": talk.get("summary", ""),
                        "key_takeaways": talk.get("key_takeaways", []),
                        "evidence_quality": talk.get("evidence_quality", ""),
                        "speaker_perspective": talk.get("speaker_perspective", ""),
                        "tools_and_projects": talk.get("tools_and_projects", []),
                    })
                else:
                    appendix.append({
                        "title": talk_title,
                        "cluster": cluster_name,
                        "summary": "",
                        "key_takeaways": [],
                        "evidence_quality": "",
                        "speaker_perspective": "",
                        "tools_and_projects": [],
                    })
        return appendix

    return []


def assemble_report(
    config: Config, outline: dict, sections: list[dict],
) -> Path:
    """Assemble final report: exec summary + Jinja2 template (Phase 4).

    Returns path to the generated report.
    """
    reports_dir = config.data_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "briefing_report.md"

    templates_dir = Path(__file__).resolve().parent.parent.parent / "templates"

    if not templates_dir.exists():
        console.print(f"{tag('report')} No templates/ directory found.")
        return out_path

    # Step A: Generate executive summary
    console.print(f"{tag('report')} Generating executive summary...")
    with console.status(f"{tag('report')} Executive summary..."):
        exec_summary = _generate_executive_summary(config, outline, sections)

    # Step B: Build appendix (no LLM)
    console.print(f"{tag('report')} Building appendix...")
    appendix_talks = _build_appendix(config, outline)
    console.print(
        f"{tag('report')} Appendix: {len(appendix_talks)} talks "
        f"(strategy: {outline.get('appendix_strategy', 'top_talks_only')})"
    )

    # Step C: Jinja2 assembly
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(templates_dir)),
        undefined=jinja2.Undefined,
        keep_trailing_newline=True,
    )

    try:
        template = env.get_template("briefing_v2.md.j2")
    except jinja2.TemplateNotFound:
        console.print(
            f"{tag('report')} [red]Template briefing_v2.md.j2 not found.[/red]"
        )
        return out_path

    context = {
        "conference_name": config.conference.name,
        "executive_summary": exec_summary,
        "sections": sections,
        "appendix_talks": appendix_talks,
        "outline": outline,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "model": config.llm.model,
    }

    content = template.render(**context)
    out_path.write_text(content)
    console.print(f"{tag('report')} Report saved to {out_path}")

    # Save metadata
    meta_path = reports_dir / "report_metadata.json"
    metadata = {
        "conference_name": config.conference.name,
        "generated_at": context["generated_at"],
        "model": config.llm.model,
        "sections": len(sections),
        "appendix_talks": len(appendix_talks),
        "outline_thesis": outline.get("thesis", ""),
    }
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    # Step D: Generate mdBook sources for `just report-book`
    _build_mdbook_src(config, outline, exec_summary, sections, appendix_talks, context)

    return out_path


# --- mdBook generation ---

_BOOK_TOML = """\
[book]
title = "{title}"
authors = ["conf-briefing"]
language = "en"
src = "src"

[output.html]
default-theme = "light"
preferred-dark-theme = "ayu"
git-repository-url = ""
"""


def _slug(text: str) -> str:
    """Convert a section title to a filename-safe slug."""
    import re

    s = text.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")[:60]


def _build_mdbook_src(
    config: Config,
    outline: dict,
    exec_summary: dict,
    sections: list[dict],
    appendix_talks: list[dict],
    context: dict,
) -> None:
    """Generate mdBook source files from report data.

    Creates:
      reports/book/book.toml
      reports/book/src/SUMMARY.md
      reports/book/src/index.md          (exec summary)
      reports/book/src/<section>.md       (one per section)
      reports/book/src/appendix.md       (curated talk list)
      reports/book/src/images/           (symlink to report images)
    """
    reports_dir = config.data_dir / "reports"
    book_dir = reports_dir / "book"
    src_dir = book_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    conf_name = config.conference.name
    generated_at = context.get("generated_at", "")
    model = context.get("model", "")
    thesis = outline.get("thesis", "")

    # book.toml
    (book_dir / "book.toml").write_text(
        _BOOK_TOML.format(title=f"Intelligence Briefing: {conf_name}")
    )

    # Symlink images into src/ so mdBook can find them
    images_link = src_dir / "images"
    if not images_link.exists():
        images_src = reports_dir / "images"
        if images_src.exists():
            images_link.symlink_to(images_src.resolve())

    # Build section files and SUMMARY entries
    summary_lines = ["# Summary\n"]
    summary_lines.append(f"- [Executive Summary](./index.md)")

    # --- index.md: executive summary ---
    index_parts = [
        f"# Intelligence Briefing: {conf_name}\n",
        f"> {thesis}\n",
        f"## Executive Summary\n",
        exec_summary.get("summary", ""),
        "\n### Key Findings\n",
    ]
    for finding in exec_summary.get("key_findings", []):
        index_parts.append(f"- {finding}")
    index_parts.append("\n### Recommended Actions\n")
    for action in exec_summary.get("top_actions", []):
        index_parts.append(f"- {action}")
    index_parts.append(f"\n---\n\n*Generated {generated_at} using {model}*")
    (src_dir / "index.md").write_text("\n".join(index_parts))

    # --- per-section files ---
    for i, section in enumerate(sections, 1):
        title = section["title"]
        slug = f"{i:02d}-{_slug(title)}"
        filename = f"{slug}.md"

        page = f"# {i}. {title}\n\n{section['prose']}\n"

        quotes = section.get("quotes", [])
        if quotes:
            page += "\n## Sources\n\n"
            for q in quotes:
                page += f"- {q['talk_title']}"
                if q.get("speaker"):
                    page += f" ({q['speaker']})"
                if q.get("timestamp_sec"):
                    m = int(q["timestamp_sec"]) // 60
                    s = int(q["timestamp_sec"]) % 60
                    page += f" at {m}:{s:02d}"
                page += "\n"

        (src_dir / filename).write_text(page)
        summary_lines.append(f"- [{i}. {title}](./{filename})")

    # --- appendix ---
    if appendix_talks:
        appendix_parts = ["# Appendix: Selected Talk Summaries\n"]
        for talk in appendix_talks:
            appendix_parts.append(f"## {talk['title']}\n")
            if talk.get("cluster"):
                appendix_parts.append(f"*Cluster: {talk['cluster']}*")
            quals = []
            if talk.get("evidence_quality"):
                quals.append(f"Evidence: {talk['evidence_quality']}")
            if talk.get("speaker_perspective"):
                quals.append(f"Perspective: {talk['speaker_perspective']}")
            if quals:
                appendix_parts.append(f" | {' | '.join(quals)}")
            if talk.get("summary"):
                appendix_parts.append(f"\n{talk['summary']}")
            if talk.get("key_takeaways"):
                appendix_parts.append("\n**Key Takeaways:**")
                for kt in talk["key_takeaways"]:
                    appendix_parts.append(f"- {kt}")
            if talk.get("tools_and_projects"):
                appendix_parts.append(
                    f"\n**Tools & Projects:** {', '.join(talk['tools_and_projects'])}"
                )
            appendix_parts.append("\n---\n")

        (src_dir / "appendix.md").write_text("\n".join(appendix_parts))
        summary_lines.append("- [Appendix: Talk Summaries](./appendix.md)")

    # --- SUMMARY.md ---
    (src_dir / "SUMMARY.md").write_text("\n".join(summary_lines) + "\n")

    console.print(
        f"{tag('report')} mdBook sources saved to {book_dir} "
        f"({len(sections)} chapters)"
    )
