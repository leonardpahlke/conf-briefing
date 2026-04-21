"""Phase 1: LLM-generated report outline from ranking + global synthesis."""

import json
from pathlib import Path

from conf_briefing.analyze.llm import query_llm_json
from conf_briefing.config import Config
from conf_briefing.console import console, tag
from conf_briefing.io import load_json_file
from conf_briefing.report.schemas import ReportOutline

SYSTEM_PROMPT = """\
You are a technical intelligence analyst creating a conference briefing \
for a cloud-native engineering team. Design a report structure that \
allocates depth based on topic relevance."""

OUTLINE_PROMPT = """\
Design the structure for an intelligence briefing report about "{conference_name}".

The report targets a cloud-native engineering team interested in these topics:
{eval_topics}

Here are the conference topic clusters, ranked by relevance (1 = most relevant):

{ranking_json}

Here is the global synthesis narrative across all clusters:

{global_narrative}

Cross-cutting themes identified:
{themes_json}

---

Produce a JSON report outline with these rules:

1. **thesis**: Write a 1-2 sentence thesis capturing the conference's main message.

2. **sections**: Create an ordered list of sections. Section types and their rules:
   - "landscape": ONE section (~600 words) painting the overall conference landscape.
   - "cluster_deep_dive": For clusters with relevance_score >= 0.70. \
Each gets its own section (~800-1200 words). Set cluster_name to the exact cluster name. \
Write guidance on what angle to cover (what's novel, what's production-proven, what's debated).
   - "cluster_brief": For clusters with relevance_score 0.40-0.69. \
Each gets a short section (~200-300 words). Set cluster_name to the exact cluster name.
   - "cross_cutting": ONE section (~500 words) on themes that span multiple clusters.
   - "tensions": ONE section (~500 words) on debates and contradictions across talks.
   - "maturity": ONE section (~400 words) on technology readiness (adopt/trial/assess/hold).
   - "actions": ONE section (~400 words) with prioritized recommendations.

   Clusters with relevance_score < 0.40 should NOT get their own section — mention them \
briefly in the landscape section instead.

3. **section_id**: Use lowercase slugs (e.g., "landscape", "cluster_ai_ml", "tensions").

4. **priority**: 1 = first in report, ascending. Order: landscape → deep dives (by rank) → \
briefs (by rank) → cross_cutting → tensions → maturity → actions.

5. **source_talks**: For cluster sections, list the recommended talk titles from the ranking. \
For thematic sections, leave empty.

6. **appendix_strategy**: Use "top_talks_only".

7. **total_word_budget**: Sum of all section word_budget values. Target ~{word_budget} words.

Use the EXACT cluster names from the ranking — do not rename them."""


def generate_outline(config: Config) -> dict:
    """Generate report outline (Phase 1).

    Returns the outline as a dict. Saves checkpoint to reports/report_outline.json.
    """
    data_dir = config.data_dir
    reports_dir = data_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "report_outline.json"

    ranking_path = data_dir / "analysis_ranking.json"
    recordings_path = data_dir / "analysis_recordings.json"

    if not ranking_path.exists() or not recordings_path.exists():
        console.print(
            f"{tag('report')} Missing ranking or global synthesis, skipping outline."
        )
        return {}

    # Cache check: skip if outline is newer than all analysis files
    if out_path.exists():
        out_mtime = out_path.stat().st_mtime
        analysis_files = [
            data_dir / f
            for f in [
                "analysis_ranking.json",
                "analysis_recordings.json",
                "analysis_clusters.json",
                "analysis_talks.json",
                "analysis_agenda.json",
            ]
            if (data_dir / f).exists()
        ]
        if all(out_mtime > f.stat().st_mtime for f in analysis_files):
            console.print(f"{tag('report')} Outline is up-to-date, skipping.")
            return load_json_file(out_path)

    ranking = load_json_file(ranking_path)
    recordings = load_json_file(recordings_path)

    eval_topics_str = "\n".join(
        f"- {t}" for t in config.analyze.eval_topics
    )

    themes_json = json.dumps(
        recordings.get("cross_cutting_themes", []), indent=2, ensure_ascii=False
    )

    prompt = OUTLINE_PROMPT.format(
        conference_name=config.conference.name,
        eval_topics=eval_topics_str,
        ranking_json=json.dumps(ranking, indent=2, ensure_ascii=False),
        global_narrative=recordings.get("narrative", ""),
        themes_json=themes_json,
        word_budget=config.report.word_budget,
    )

    console.print(f"{tag('report')} Generating report outline...")
    with console.status(f"{tag('report')} Outline generation..."):
        result = query_llm_json(
            config, SYSTEM_PROMPT, prompt, max_tokens=8192, schema=ReportOutline
        )

    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    console.print(f"{tag('report')} Outline saved to {out_path}")
    return result
