"""Phase 2: Per-section LLM drafting with parallel execution."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from conf_briefing.analyze.llm import query_llm_json
from conf_briefing.config import Config
from conf_briefing.console import console, progress_bar, tag
from conf_briefing.io import load_json_file
from conf_briefing.report.schemas import SectionDraft

SYSTEM_PROMPT = """\
You are a technical intelligence analyst writing a conference briefing section \
for a cloud-native engineering team. Write analytical prose, not bullet lists. \
Reference specific talks and speakers by name. Distinguish production evidence \
from theoretical claims or vendor demos."""

# --- Per-section-type prompts ---

DEEP_DIVE_PROMPT = """\
Write a deep-dive analysis section for the topic cluster "{title}" from {conference_name}.

{guidance}

Here is the cluster synthesis (narrative, themes, tensions, technologies):
{cluster_synthesis}

Here are the individual talk analyses for this cluster:
{talks_json}

Write ~{word_budget} words of analytical markdown prose. Requirements:
- Open with the key narrative thread connecting these talks.
- Reference specific talks by title and speakers by name.
- Highlight where production evidence differs from theoretical claims.
- Call out the 2-3 most important talks worth watching and why.
- End with implications for practitioners.

Also return:
- citations: For each specific claim that references a talk, include a citation. \
Set needs_quote=true for quantitative claims, direct speaker attributions, or \
strong assertions that would benefit from a verbatim transcript quote.
- key_takeaway: A single sentence capturing this section's main insight."""

BRIEF_PROMPT = """\
Write a brief overview for the topic cluster "{title}" from {conference_name}.

Cluster synthesis:
{cluster_synthesis}

Recommended talks: {recommended_talks}

Write ~{word_budget} words as 1-2 paragraphs of analytical markdown prose. Name the key \
technology or trend and the top talk worth watching.

Return citations (needs_quote=false for briefs) and a key_takeaway sentence."""

LANDSCAPE_PROMPT = """\
Write a conference landscape overview for {conference_name}.

Conference narrative:
{narrative}

Top keywords: {keywords}

Track distribution:
{track_distribution}

Company presence (top 20):
{company_presence}

Total talks analyzed: {talk_count}
Number of topic clusters: {cluster_count}

Write ~{word_budget} words of analytical markdown prose. Paint the overall landscape: \
what topics dominated, which companies were most active, how tracks were distributed, \
and what the conference's center of gravity was. Do not use bullet lists.

Return an empty citations list and a key_takeaway sentence."""

CROSS_CUTTING_PROMPT = """\
Write a cross-cutting themes section for {conference_name}.

Themes that span multiple clusters:
{themes_json}

Write ~{word_budget} words of analytical markdown prose. For each theme, explain how \
it manifests across different topic areas and why it matters. Draw connections \
between seemingly unrelated clusters.

Return citations for claims referencing specific talks (needs_quote=false) and a key_takeaway."""

TENSIONS_PROMPT = """\
Write a tensions and debates section for {conference_name}.

Tensions identified across clusters:
{tensions_json}

Write ~{word_budget} words of analytical markdown prose. Frame each tension as a genuine \
debate with legitimate arguments on both sides. Note the severity and practical \
implications for practitioners who must choose a side.

Return citations (needs_quote=true for strong opposing claims) and a key_takeaway."""

MATURITY_PROMPT = """\
Write a technology maturity assessment section for {conference_name}.

Maturity landscape (technology radar):
{maturity_json}

Write ~{word_budget} words of analytical markdown prose. Organize by maturity ring \
(Adopt, Trial, Assess, Hold). For each ring, explain which technologies belong there \
and why based on the evidence quality. This section should help practitioners decide \
what to invest in now vs. watch for later.

Return citations (needs_quote=false) and a key_takeaway."""

ACTIONS_PROMPT = """\
Write a recommended actions section for {conference_name}.

Recommended actions synthesized across all clusters:
{actions_json}

Evaluation topics the team cares about:
{eval_topics}

Write ~{word_budget} words of analytical markdown prose. Group actions by urgency \
(immediate, next quarter, long-term). For each action, state what to do, why, and \
what evidence supports it. Be specific and actionable.

Return citations (needs_quote=false) and a key_takeaway."""


def _load_analysis_data(config: Config) -> dict:
    """Load all analysis files needed for drafting."""
    data_dir = config.data_dir
    return {
        "agenda": load_json_file(data_dir / "analysis_agenda.json")
        if (data_dir / "analysis_agenda.json").exists()
        else {},
        "ranking": load_json_file(data_dir / "analysis_ranking.json")
        if (data_dir / "analysis_ranking.json").exists()
        else [],
        "clusters": load_json_file(data_dir / "analysis_clusters.json")
        if (data_dir / "analysis_clusters.json").exists()
        else [],
        "talks": load_json_file(data_dir / "analysis_talks.json")
        if (data_dir / "analysis_talks.json").exists()
        else [],
        "recordings": load_json_file(data_dir / "analysis_recordings.json")
        if (data_dir / "analysis_recordings.json").exists()
        else {},
    }


def _condense_talk(talk: dict) -> dict:
    """Trim a talk analysis to essential fields for section drafting."""
    keep = [
        "title", "summary", "key_takeaways", "tools_and_projects",
        "problems_discussed", "evidence_quality", "speaker_perspective",
        "maturity_assessments", "caveats_and_concerns", "technology_stance",
    ]
    return {k: talk[k] for k in keep if k in talk}


def _find_cluster_synthesis(clusters: list[dict], cluster_name: str) -> dict:
    """Find cluster synthesis by name."""
    for cs in clusters:
        if cs.get("cluster_name") == cluster_name:
            return cs
    return {}


def _find_cluster_talks(
    agenda: dict, talks: list[dict], cluster_name: str,
) -> list[dict]:
    """Find and condense talk analyses belonging to a cluster."""
    # Get talk titles from agenda cluster
    cluster_talk_titles = set()
    for cluster in agenda.get("clusters", []):
        if cluster.get("name") == cluster_name:
            cluster_talk_titles = set(cluster.get("talks", []))
            break

    talks_by_title = {t["title"]: t for t in talks if t.get("title")}
    # Prefix matching for fuzzy title resolution
    talks_by_prefix = {}
    for title, talk in talks_by_title.items():
        prefix = title.split(" - ")[0].strip()
        if prefix not in talks_by_prefix:
            talks_by_prefix[prefix] = talk

    result = []
    for title in cluster_talk_titles:
        talk = talks_by_title.get(title)
        if not talk:
            prefix = title.split(" - ")[0].strip()
            talk = talks_by_prefix.get(prefix)
        if not talk:
            for full_title, t in talks_by_title.items():
                if full_title.startswith(title):
                    talk = t
                    break
        if talk:
            result.append(_condense_talk(talk))
    return result


def _find_recommended_talks(ranking: list[dict], cluster_name: str) -> list[str]:
    """Get recommended talk titles for a cluster from the ranking."""
    for entry in ranking:
        if entry.get("name") == cluster_name:
            return entry.get("recommended_talks", [])
    return []


def _extract_company_name(role_string: str) -> str:
    """Extract company name from role strings like 'Software Engineer, Google'."""
    if not role_string:
        return role_string
    # Common pattern: "Title, Company" or just "Company"
    parts = role_string.rsplit(", ", 1)
    if len(parts) == 2:
        return parts[1].strip()
    return role_string.strip()


def _draft_section(
    config: Config, section: dict, data: dict,
) -> dict | None:
    """Draft a single section via LLM call."""
    section_type = section["section_type"]
    title = section["title"]
    cluster_name = section.get("cluster_name", "")
    word_budget = section.get("word_budget", 500)
    guidance = section.get("guidance", "")
    conf_name = config.conference.name

    if section_type == "cluster_deep_dive":
        cs = _find_cluster_synthesis(data["clusters"], cluster_name)
        cluster_talks = _find_cluster_talks(
            data["agenda"], data["talks"], cluster_name
        )
        prompt = DEEP_DIVE_PROMPT.format(
            title=title,
            conference_name=conf_name,
            guidance=f"Editorial guidance: {guidance}" if guidance else "",
            cluster_synthesis=json.dumps(cs, indent=2, ensure_ascii=False),
            talks_json=json.dumps(cluster_talks, indent=2, ensure_ascii=False),
            word_budget=word_budget,
        )
        max_tokens = 6144

    elif section_type == "cluster_brief":
        cs = _find_cluster_synthesis(data["clusters"], cluster_name)
        rec_talks = _find_recommended_talks(data["ranking"], cluster_name)
        prompt = BRIEF_PROMPT.format(
            title=title,
            conference_name=conf_name,
            cluster_synthesis=json.dumps(
                {k: cs[k] for k in ["narrative", "cross_cutting_themes"] if k in cs},
                indent=2, ensure_ascii=False,
            ),
            recommended_talks=", ".join(rec_talks) if rec_talks else "None specified",
            word_budget=word_budget,
        )
        max_tokens = 2048

    elif section_type == "landscape":
        agenda = data["agenda"]
        # Clean company presence
        raw_presence = agenda.get("company_presence", {})
        clean_presence = {}
        for role_str, count in raw_presence.items():
            company = _extract_company_name(role_str)
            clean_presence[company] = clean_presence.get(company, 0) + count
        # Sort and take top 20
        sorted_companies = sorted(
            clean_presence.items(), key=lambda x: x[1], reverse=True
        )[:20]

        prompt = LANDSCAPE_PROMPT.format(
            conference_name=conf_name,
            narrative=agenda.get("narrative", ""),
            keywords=", ".join(agenda.get("top_keywords", [])[:20]),
            track_distribution=json.dumps(
                agenda.get("track_distribution", {}), indent=2
            ),
            company_presence=json.dumps(
                dict(sorted_companies), indent=2
            ),
            talk_count=len(data["talks"]),
            cluster_count=len(agenda.get("clusters", [])),
            word_budget=word_budget,
        )
        max_tokens = 4096

    elif section_type == "cross_cutting":
        recordings = data["recordings"]
        prompt = CROSS_CUTTING_PROMPT.format(
            conference_name=conf_name,
            themes_json=json.dumps(
                recordings.get("cross_cutting_themes", []),
                indent=2, ensure_ascii=False,
            ),
            word_budget=word_budget,
        )
        max_tokens = 4096

    elif section_type == "tensions":
        recordings = data["recordings"]
        prompt = TENSIONS_PROMPT.format(
            conference_name=conf_name,
            tensions_json=json.dumps(
                recordings.get("tensions", []),
                indent=2, ensure_ascii=False,
            ),
            word_budget=word_budget,
        )
        max_tokens = 4096

    elif section_type == "maturity":
        recordings = data["recordings"]
        prompt = MATURITY_PROMPT.format(
            conference_name=conf_name,
            maturity_json=json.dumps(
                recordings.get("maturity_landscape", []),
                indent=2, ensure_ascii=False,
            ),
            word_budget=word_budget,
        )
        max_tokens = 4096

    elif section_type == "actions":
        recordings = data["recordings"]
        prompt = ACTIONS_PROMPT.format(
            conference_name=conf_name,
            actions_json=json.dumps(
                recordings.get("recommended_actions", []),
                indent=2, ensure_ascii=False,
            ),
            eval_topics="\n".join(f"- {t}" for t in config.analyze.eval_topics),
            word_budget=word_budget,
        )
        max_tokens = 4096
    else:
        console.print(
            f"  {tag('report')} Unknown section type '{section_type}', skipping."
        )
        return None

    result = query_llm_json(
        config, SYSTEM_PROMPT, prompt, max_tokens=max_tokens, schema=SectionDraft
    )

    # Overwrite section_id and title from outline (avoid LLM paraphrasing)
    result["section_id"] = section["section_id"]
    result["title"] = title

    return result


def draft_sections(config: Config, outline: dict) -> list[dict]:
    """Draft all report sections in parallel (Phase 2).

    Returns list of section draft dicts. Saves checkpoint incrementally.
    """
    data_dir = config.data_dir
    reports_dir = data_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "report_sections.json"
    outline_path = reports_dir / "report_outline.json"

    sections = outline.get("sections", [])
    if not sections:
        console.print(f"{tag('report')} No sections in outline, skipping drafting.")
        return []

    # Cache check
    if out_path.exists() and outline_path.exists():
        if out_path.stat().st_mtime > outline_path.stat().st_mtime:
            console.print(f"{tag('report')} Section drafts are up-to-date, skipping.")
            return load_json_file(out_path)

    data = _load_analysis_data(config)

    console.print(
        f"{tag('report')} Drafting {len(sections)} sections..."
    )

    drafts: list[dict] = []
    failed = 0

    with progress_bar() as pb:
        task = pb.add_task(
            f"{tag('report')} Drafting sections", total=len(sections)
        )

        with ThreadPoolExecutor(max_workers=config.llm.num_parallel) as executor:
            futures = {
                executor.submit(_draft_section, config, s, data): s
                for s in sections
            }
            for future in as_completed(futures):
                section = futures[future]
                title = section["title"][:50]
                try:
                    result = future.result()
                    if result:
                        drafts.append(result)
                        # Save incrementally
                        out_path.write_text(
                            json.dumps(drafts, indent=2, ensure_ascii=False)
                        )
                    pb.update(task, advance=1, description=f"{tag('report')} {title}")
                except Exception as e:
                    failed += 1
                    pb.update(
                        task, advance=1,
                        description=f"{tag('report')} {title} [red]failed[/red]",
                    )
                    console.print(f"  {tag('report')} [red]{title} — {e}[/red]")

    if failed:
        console.print(
            f"{tag('report')} [yellow]{failed}/{len(sections)} sections failed[/yellow]"
        )

    # Sort by outline priority
    section_order = {s["section_id"]: s.get("priority", 99) for s in sections}
    drafts.sort(key=lambda d: section_order.get(d["section_id"], 99))

    out_path.write_text(json.dumps(drafts, indent=2, ensure_ascii=False))
    console.print(
        f"{tag('report')} {len(drafts)} section drafts saved to {out_path}"
    )
    return drafts
