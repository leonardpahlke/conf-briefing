"""Agenda analysis: cluster talks from rich content (Phase 2)."""

import json
from collections import Counter
from pathlib import Path

from conf_briefing.analyze.llm import query_llm_json
from conf_briefing.analyze.schemas import AgendaClustering
from conf_briefing.config import Config
from conf_briefing.console import console, tag
from conf_briefing.io import load_json_file

def _extract_company_name(role_string: str) -> str:
    """Extract company name from role strings like 'Software Engineer, Google'.

    Handles patterns:
      "Software Engineer, Google" → "Google"
      "CTO & Co-Founder, Robusta.dev, Robusta.dev" → "Robusta.dev"
      "Akamai" → "Akamai"
      "(Independent)" → "Independent"
      "" → ""
    """
    if not role_string:
        return ""
    s = role_string.strip().strip("()")
    if not s:
        return ""
    # If there's a comma, the company is typically the last part
    if ", " in s:
        return s.rsplit(", ", 1)[1].strip()
    return s


SYSTEM_PROMPT = """\
You are a conference analysis expert. You analyze conference talks to identify \
themes, topics, and trends. Always respond with valid JSON."""

CLUSTER_PROMPT_TEMPLATE = """\
Analyze the following conference talk analyses for "{conference_name}" and group them \
into thematic clusters based on shared technologies, problem domains, and methodologies.

Talk analyses (each includes summary, key takeaways, tools, and problems):
{talks_json}

Produce a JSON object with these fields:
{{
  "clusters": [
    {{
      "name": "cluster name",
      "description": "brief description of what unites these talks",
      "talks": ["talk title 1", "talk title 2", ...],
      "keywords": ["keyword1", "keyword2", ...],
      "companies": ["company1", "company2", ...]
    }}
  ],
  "top_keywords": ["keyword1", "keyword2", ...],
  "narrative": "A 2-3 paragraph markdown summary of the conference landscape."
}}

Group talks into 15-25 thematic clusters, each containing 10-25 talks. \
Cluster by shared technologies, problem domains, and methodologies revealed in the analyses. \
Ensure every talk appears in at least one cluster. \
Name each cluster with a descriptive noun phrase capturing the shared problem domain or \
technology area (e.g. "eBPF-Based Networking", "Platform Engineering at Scale", "Supply Chain Security"). \
Extract the top 20 keywords."""


def cluster_talks(config: Config) -> Path | None:
    """Cluster talks from rich analysis content (Phase 2).

    Input: analysis_talks.json (per-talk analyses) enriched with track/speakers
    from schedule_clean.json.
    Output: analysis_agenda.json (clusters, keywords, company presence, etc.)
    Caching: skip if analysis_agenda.json mtime > analysis_talks.json mtime.
    """
    data_dir = config.data_dir
    talks_path = data_dir / "analysis_talks.json"
    schedule_path = data_dir / "schedule_clean.json"
    out_path = data_dir / "analysis_agenda.json"

    if not talks_path.exists():
        console.print(f"{tag('analyze')} No talk analyses found, skipping clustering.")
        return None

    # Cache check: skip if output is newer than input
    if out_path.exists() and out_path.stat().st_mtime > talks_path.stat().st_mtime:
        console.print(f"{tag('analyze')} Clustering is up-to-date, skipping.")
        return out_path

    talks = load_json_file(talks_path)

    # Enrich with track/speakers from schedule if available
    schedule_by_title: dict[str, dict] = {}
    if schedule_path.exists():
        schedule = load_json_file(schedule_path)
        schedule_by_title = {s["title"]: s for s in schedule if s.get("title")}

    # Condense to fields useful for clustering
    condensed = []
    for talk in talks:
        entry = {
            "title": talk.get("title", ""),
            "summary": talk.get("summary", ""),
            "key_takeaways": talk.get("key_takeaways", []),
            "tools_and_projects": talk.get("tools_and_projects", []),
            "problems_discussed": talk.get("problems_discussed", []),
            "evidence_quality": talk.get("evidence_quality", ""),
            "speaker_perspective": talk.get("speaker_perspective", ""),
        }
        # Enrich with schedule metadata
        sched = schedule_by_title.get(talk.get("title", ""), {})
        if sched.get("track"):
            entry["track"] = sched["track"]
        if sched.get("speakers"):
            entry["speakers"] = sched["speakers"]
        condensed.append(entry)

    # Compute company_presence and track_distribution programmatically
    company_counter: Counter[str] = Counter()
    track_counter: Counter[str] = Counter()
    for entry in condensed:
        track = entry.get("track", "")
        if track:
            track_counter[track] += 1
        for speaker in entry.get("speakers", []):
            company = _extract_company_name(speaker.get("company", ""))
            if company:
                company_counter[company] += 1

    prompt = CLUSTER_PROMPT_TEMPLATE.format(
        conference_name=config.conference.name,
        talks_json=json.dumps(condensed, indent=2, ensure_ascii=False),
    )

    console.print(f"{tag('analyze')} Clustering {len(talks)} analyzed talks...")
    with console.status(f"{tag('analyze')} Clustering talks..."):
        result = query_llm_json(
            config, SYSTEM_PROMPT, prompt, max_tokens=16384, schema=AgendaClustering
        )

    # Merge programmatically-computed counts into result
    result["company_presence"] = dict(company_counter.most_common())
    result["track_distribution"] = dict(track_counter.most_common())

    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    console.print(f"{tag('analyze')} Clustering saved to {out_path}")
    return out_path
