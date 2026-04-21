"""Phase 4: Executive summary generation + final Jinja2 assembly."""

import json
import re
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
Be specific — include technology names and company examples. Only cite a metric \
(percentage, number) if you can attribute it to the SAME section and technology \
it appears with in the data. Do NOT combine a metric from one section with a \
technology from a different section. Do NOT repeat the same metric for different \
technologies — use distinct evidence for each finding.
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

    # Build structured per-section summaries (preserves metric attribution)
    section_summaries = []
    for s in sections:
        kt = s.get("key_takeaway", "")
        prose_preview = s["prose"][:500]
        section_summaries.append(
            f"## {s['title']}\n"
            f"Key takeaway: {kt}\n"
            f"Preview: {prose_preview}..."
        )
    body_preview = "\n\n---\n\n".join(section_summaries)

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


# Charts to embed in specific sections
_SECTION_CHARTS: dict[str, list[tuple[str, str]]] = {
    "landscape": [
        ("images/track_distribution.svg", "Track Distribution"),
        ("images/company_presence.svg", "Company Presence"),
        ("images/topic_frequency.svg", "Top Conference Topics"),
    ],
    "maturity": [
        ("images/maturity_strip.svg", "Technology Maturity Landscape"),
    ],
}

# Role-based reading guide: role → list of section_id keywords
_ROLE_SECTIONS: dict[str, list[str]] = {
    "Platform Engineers": [
        "cluster_platform", "cluster_operators", "cluster_scheduling", "actions",
    ],
    "SREs & Operations": [
        "cluster_observability", "cluster_ai_ml", "maturity", "actions",
    ],
    "Security Leads": [
        "cluster_security", "cluster_supply_chain", "tensions",
    ],
}

_BRIEF_MERGE_THRESHOLD = 2500  # chars — briefs shorter than this get merged


def _promote_bold_to_headings(prose: str) -> str:
    """Promote **Bold text** at paragraph starts to ### headings."""
    return re.sub(
        r"\n\n\*\*([^*]+)\*\*\s*\n",
        r"\n\n### \1\n\n",
        prose,
    )


def _add_crossrefs(
    page: str,
    appendix_anchors: dict[str, str],
    section_links: dict[str, str],
    is_synthesis: bool,
) -> str:
    """Add links to appendix entries and other chapters in a section page."""
    # Link talk titles to appendix entries
    for talk_title, anchor in appendix_anchors.items():
        escaped = re.escape(talk_title)
        # **Talk Title** → **[Talk Title](appendix.md#anchor)**
        page = re.sub(
            rf"\*\*({escaped})\*\*",
            f"**[\\1]({anchor})**",
            page,
        )
        # *Talk Title* → *[Talk Title](appendix.md#anchor)*
        page = re.sub(
            rf"(?<!\*)\*({escaped})\*(?!\*)",
            f"*[\\1]({anchor})*",
            page,
        )

    # In synthesis sections, link cluster/section name mentions to chapters
    if is_synthesis:
        for sec_title, target in section_links.items():
            escaped = re.escape(sec_title)
            page = re.sub(
                rf"\*({escaped})\*",
                f"*[\\1]({target})*",
                page,
            )

    return page


def _build_mdbook_src(
    config: Config,
    outline: dict,
    exec_summary: dict,
    sections: list[dict],
    appendix_talks: list[dict],
    context: dict,
) -> None:
    """Generate mdBook source files from report data."""
    reports_dir = config.data_dir / "reports"
    book_dir = reports_dir / "book"
    src_dir = book_dir / "src"
    # Clean stale .md files from previous runs (preserve images symlink)
    if src_dir.exists():
        for old_file in src_dir.glob("*.md"):
            old_file.unlink()
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

    # --- Classify sections: regular vs thin briefs (1B) ---
    outline_specs = {
        s["section_id"]: s for s in outline.get("sections", [])
    }
    regular_sections: list[dict] = []
    thin_briefs: list[dict] = []
    for section in sections:
        sid = section.get("section_id", "")
        spec = outline_specs.get(sid, {})
        is_brief = spec.get("section_type") == "cluster_brief"
        is_thin = len(section.get("prose", "")) < _BRIEF_MERGE_THRESHOLD
        if is_brief and is_thin:
            thin_briefs.append(section)
        else:
            regular_sections.append(section)

    summary_lines = ["# Summary\n"]
    summary_lines.append("- [Executive Summary](./index.md)")

    # --- index.md: executive summary ---
    index_parts = [
        f"# Intelligence Briefing: {conf_name}\n",
        f"> {thesis}\n",
        "## Executive Summary\n",
        exec_summary.get("summary", ""),
        "\n### Key Findings\n",
    ]
    for finding in exec_summary.get("key_findings", []):
        index_parts.append(f"- {finding}")
    index_parts.append("\n### Recommended Actions\n")
    for action in exec_summary.get("top_actions", []):
        index_parts.append(f"- {action}")

    # Chart: cluster relevance overview in exec summary (1A)
    images_src = reports_dir / "images"
    if (images_src / "cluster_relevance.svg").exists():
        index_parts.append("\n### Topic Relevance\n")
        index_parts.append("![Cluster Relevance](./images/cluster_relevance.svg)")

    # Reading guide (1E)
    section_file_map: dict[str, tuple[int, str, str]] = {}  # sid → (num, title, filename)
    for i, sec in enumerate(regular_sections, 1):
        sid = sec.get("section_id", "")
        slug = f"{i:02d}-{_slug(sec['title'])}"
        section_file_map[sid] = (i, sec["title"], f"{slug}.md")

    index_parts.append("\n### Reading Guide\n")
    for role, role_sids in _ROLE_SECTIONS.items():
        links = []
        for rsid in role_sids:
            info = section_file_map.get(rsid)
            if info:
                num, title, fname = info
                links.append(f"[{num}. {title}](./{fname})")
        if links:
            index_parts.append(f"- **{role}**: {', '.join(links)}")
    # Short-on-time suggestion
    cross_cut = section_file_map.get("cross_cutting")
    actions_sec = section_file_map.get("actions")
    if cross_cut and actions_sec:
        index_parts.append(
            f"- **Short on time?** Read this summary, then "
            f"[{cross_cut[0]}. {cross_cut[1]}](./{cross_cut[2]}) and "
            f"[{actions_sec[0]}. {actions_sec[1]}](./{actions_sec[2]})"
        )

    index_parts.append(f"\n---\n\n*Generated {generated_at} using {model}*")
    (src_dir / "index.md").write_text("\n".join(index_parts))

    # --- Build lookups for cross-references (1D) ---
    appendix_anchors: dict[str, str] = {}
    if appendix_talks:
        for talk in appendix_talks:
            title = talk["title"]
            appendix_anchors[title] = f"./appendix.md#{_slug(title)}"

    section_links: dict[str, str] = {}
    for _, title, fname in section_file_map.values():
        section_links[title] = f"./{fname}"

    synthesis_ids = {"cross_cutting", "tensions", "maturity", "actions"}

    # --- Per-section files ---
    for i, section in enumerate(regular_sections, 1):
        title = section["title"]
        sid = section.get("section_id", "")
        slug = f"{i:02d}-{_slug(title)}"
        filename = f"{slug}.md"

        # 1C: promote bold openers to subheadings
        prose = _promote_bold_to_headings(section["prose"])

        page = f"# {i}. {title}\n\n{prose}\n"

        # 1A: embed charts
        charts = _SECTION_CHARTS.get(sid, [])
        if charts:
            for chart_path, caption in charts:
                if (images_src / Path(chart_path).name).exists():
                    page += f"\n![{caption}](./{chart_path})\n"

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

        # 1D: cross-references
        page = _add_crossrefs(
            page, appendix_anchors, section_links,
            is_synthesis=sid in synthesis_ids,
        )

        (src_dir / filename).write_text(page)
        summary_lines.append(f"- [{i}. {title}](./{filename})")

    # --- Merged brief notes (1B) ---
    if thin_briefs:
        brief_num = len(regular_sections) + 1
        brief_filename = f"{brief_num:02d}-brief-notes.md"
        brief_parts = [
            "# Brief Notes\n",
            "The following topics had limited conference coverage "
            "but are noted for awareness.\n",
        ]
        for bf in thin_briefs:
            prose = _promote_bold_to_headings(bf["prose"])
            brief_parts.append(f"## {bf['title']}\n\n{prose}\n\n---\n")

        brief_page = "\n".join(brief_parts)
        brief_page = _add_crossrefs(
            brief_page, appendix_anchors, section_links,
            is_synthesis=False,
        )
        (src_dir / brief_filename).write_text(brief_page)
        summary_lines.append(f"- [{brief_num}. Brief Notes](./{brief_filename})")

    # --- Appendix ---
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
                    f"\n**Tools & Projects:** "
                    f"{', '.join(talk['tools_and_projects'])}"
                )
            appendix_parts.append("\n---\n")

        (src_dir / "appendix.md").write_text("\n".join(appendix_parts))
        summary_lines.append("- [Appendix: Talk Summaries](./appendix.md)")

    # --- SUMMARY.md ---
    (src_dir / "SUMMARY.md").write_text("\n".join(summary_lines) + "\n")

    merged_note = f" ({len(thin_briefs)} merged into Brief Notes)" if thin_briefs else ""
    console.print(
        f"{tag('report')} mdBook sources saved to {book_dir} "
        f"({len(regular_sections)} chapters{merged_note})"
    )
