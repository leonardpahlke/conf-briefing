"""Cluster ranking: summarize, argue relevance, rank clusters."""

import json
from pathlib import Path

from conf_briefing.analyze.llm import query_llm_json
from conf_briefing.config import Config
from conf_briefing.console import console, tag
from conf_briefing.io import load_json_file

SYSTEM_PROMPT = """\
You are a conference analysis expert. You rank and evaluate topic clusters \
based on relevance to specific focus areas. Always respond with valid JSON."""

RANKING_PROMPT_TEMPLATE = """\
Given the following conference topic clusters, rank them by relevance.

Conference: "{conference_name}"

Evaluation criteria:
{eval_topics}

Clusters:
{clusters_json}

For each cluster, produce a JSON array sorted by relevance (most relevant first):
[
  {{
    "name": "cluster name",
    "rank": 1,
    "relevance_score": 0.0-1.0,
    "summary": "2-3 sentence summary of this cluster",
    "relevance_argument": "Why this cluster matters",
    "recommended_talks": ["talk title 1", "talk title 2", ...],
    "skip_reason": null or "reason to skip if low relevance"
  }}
]

Be specific about why each cluster is or isn't relevant. \
Recommend the top 3-5 talks per cluster.

Score calibration: distribute relevance_score across the full 0.0-1.0 range. \
The most relevant cluster should score above 0.9 and the least relevant below 0.3. \
Avoid clustering all scores in the 0.6-0.9 range."""


def rank_clusters(config: Config) -> Path | None:
    """Rank agenda clusters by relevance."""
    data_dir = config.data_dir
    agenda_path = data_dir / "analysis_agenda.json"
    out_path = data_dir / "analysis_ranking.json"

    if not agenda_path.exists():
        console.print(f"{tag('analyze')} No agenda analysis found, skipping ranking.")
        return None

    # Cache check: skip if output is newer than input
    if out_path.exists() and out_path.stat().st_mtime > agenda_path.stat().st_mtime:
        console.print(f"{tag('analyze')} Cluster ranking is up-to-date, skipping.")
        return out_path

    agenda = load_json_file(agenda_path)
    clusters = agenda.get("clusters", [])

    if not clusters:
        console.print(f"{tag('analyze')} No clusters found in agenda analysis, skipping ranking.")
        return None

    eval_topics = config.analyze.eval_topics

    prompt = RANKING_PROMPT_TEMPLATE.format(
        conference_name=config.conference.name,
        eval_topics="\n".join(f"- {t}" for t in eval_topics),
        clusters_json=json.dumps(clusters, indent=2, ensure_ascii=False),
    )

    console.print(f"{tag('analyze')} Ranking {len(clusters)} clusters...")
    with console.status(f"{tag('analyze')} Ranking clusters..."):
        result = query_llm_json(config, SYSTEM_PROMPT, prompt, max_tokens=8192)

    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    console.print(f"{tag('analyze')} Cluster ranking saved to {out_path}")
    return out_path
