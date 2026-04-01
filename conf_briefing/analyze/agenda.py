"""Agenda analysis: cluster talks, extract topics/keywords/companies."""

import json
from pathlib import Path

from conf_briefing.analyze.llm import query_llm_json
from conf_briefing.config import Config
from conf_briefing.console import console, tag

SYSTEM_PROMPT = """\
You are a conference analysis expert. You analyze conference schedules to identify \
themes, topics, and trends. Always respond with valid JSON."""

AGENDA_PROMPT_TEMPLATE = """\
Analyze the following conference schedule for "{conference_name}".

Schedule data (JSON):
{schedule_json}

Produce a JSON object with these fields:
{{
  "clusters": [
    {{
      "name": "cluster name",
      "description": "brief description",
      "talks": ["talk title 1", "talk title 2", ...],
      "keywords": ["keyword1", "keyword2", ...],
      "companies": ["company1", "company2", ...]
    }}
  ],
  "top_keywords": ["keyword1", "keyword2", ...],
  "company_presence": {{
    "company name": <number of talks>
  }},
  "track_distribution": {{
    "track name": <number of talks>
  }},
  "narrative": "A 2-3 paragraph markdown summary of the conference landscape."
}}

Group talks into 15-25 thematic clusters, each containing 10-25 talks. \
Ensure every session appears in at least one cluster. \
Extract the top 20 keywords. \
Count company mentions across speakers. Summarize track distribution."""


def analyze_agenda(config: Config) -> Path:
    """Analyze the conference agenda using LLM."""
    data_dir = config.data_dir
    schedule_path = data_dir / "schedule_clean.json"
    out_path = data_dir / "analysis_agenda.json"

    if not schedule_path.exists():
        console.print(f"{tag('analyze')} No cleaned schedule found, skipping agenda analysis.")
        return out_path

    sessions = json.loads(schedule_path.read_text())
    # Send condensed version (titles + abstracts + speakers) to fit in context
    condensed = [
        {
            "title": s["title"],
            "abstract": s["abstract"][:300],
            "speakers": s["speakers"],
            "track": s["track"],
        }
        for s in sessions
    ]

    prompt = AGENDA_PROMPT_TEMPLATE.format(
        conference_name=config.conference.name,
        schedule_json=json.dumps(condensed, indent=2, ensure_ascii=False),
    )

    console.print(f"{tag('analyze')} Running agenda analysis on {len(sessions)} sessions...")
    with console.status(f"{tag('analyze')} Analyzing agenda..."):
        result = query_llm_json(config, SYSTEM_PROMPT, prompt, max_tokens=16384)

    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    console.print(f"{tag('analyze')} Agenda analysis saved to {out_path}")
    return out_path
