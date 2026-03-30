"""Cluster ranking: summarize, argue relevance, rank clusters."""

import json
from pathlib import Path

from conf_briefing.analyze.llm import query_llm_json
from conf_briefing.config import Config

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
Recommend the top 3-5 talks per cluster."""


def rank_clusters(config: Config) -> Path:
    """Rank agenda clusters by relevance."""
    data_dir = config.data_dir
    agenda_path = data_dir / "analysis_agenda.json"
    out_path = data_dir / "analysis_ranking.json"

    if not agenda_path.exists():
        print("[analyze] No agenda analysis found, skipping ranking.")
        return out_path

    agenda = json.loads(agenda_path.read_text())
    clusters = agenda.get("clusters", [])

    if not clusters:
        print("[analyze] No clusters found in agenda analysis, skipping ranking.")
        return out_path

    eval_topics = ["general industry trends", "emerging technology", "practical applicability"]

    prompt = RANKING_PROMPT_TEMPLATE.format(
        conference_name=config.conference.name,
        eval_topics="\n".join(f"- {t}" for t in eval_topics),
        clusters_json=json.dumps(clusters, indent=2, ensure_ascii=False),
    )

    print(f"[analyze] Ranking {len(clusters)} clusters...")
    result = query_llm_json(config, SYSTEM_PROMPT, prompt, max_tokens=8192)

    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[analyze] Cluster ranking saved to {out_path}")
    return out_path
