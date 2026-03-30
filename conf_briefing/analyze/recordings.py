"""Recording analysis: summarize transcripts, extract insights."""

import json
from pathlib import Path

from conf_briefing.analyze.llm import query_llm_json
from conf_briefing.config import Config
from conf_briefing.console import console, progress_bar, tag

SYSTEM_PROMPT = """\
You are a conference talk analyst. You extract key insights from talk transcripts. \
Always respond with valid JSON unless instructed otherwise."""

SINGLE_TALK_PROMPT = """\
Analyze this conference talk transcript.

Title: {title}
Speakers: {speakers}
Track: {track}

Transcript:
{transcript}

Produce a JSON object:
{{
  "title": "{title}",
  "key_takeaways": ["takeaway 1", "takeaway 2", ...],
  "qa_highlights": ["question/answer 1", ...],
  "problems_discussed": ["problem 1", ...],
  "tools_and_projects": ["tool/project 1", ...],
  "emerging_signals": ["signal 1", ...],
  "summary": "3-5 sentence summary"
}}"""

SYNTHESIS_PROMPT = """\
Synthesize insights across these conference talk analyses.

Conference: "{conference_name}"
Focus: {focus_topics}

Individual talk analyses:
{analyses_json}

Produce a JSON object:
{{
  "cross_cutting_themes": [
    {{
      "theme": "theme name",
      "description": "how this theme appears across talks",
      "supporting_talks": ["talk title 1", ...]
    }}
  ],
  "emerging_technologies": [
    {{
      "technology": "name",
      "mentions": <count>,
      "context": "how it was discussed"
    }}
  ],
  "common_problems": ["problem 1", ...],
  "narrative": "4-6 paragraph markdown synthesis of key insights across all talks."
}}"""


def analyze_single_talk(config: Config, session: dict) -> dict | None:
    """Analyze a single talk transcript."""
    transcript = session.get("transcript", "")
    if not transcript or len(transcript) < 100:
        return None

    speakers = ", ".join(
        f"{s['name']} ({s['company']})" if s.get("company") else s["name"]
        for s in session.get("speakers", [])
    ) or "Unknown"

    prompt = SINGLE_TALK_PROMPT.format(
        title=session["title"],
        speakers=speakers,
        track=session.get("track", ""),
        transcript=transcript[:15000],  # Cap at ~15K chars to stay within context
    )

    return query_llm_json(config, SYSTEM_PROMPT, prompt, max_tokens=4096)


def synthesize_analyses(config: Config, analyses: list[dict]) -> dict:
    """Synthesize insights across all talk analyses."""
    focus = ["general industry trends"]

    prompt = SYNTHESIS_PROMPT.format(
        conference_name=config.conference.name,
        focus_topics=", ".join(focus),
        analyses_json=json.dumps(analyses, indent=2, ensure_ascii=False),
    )

    return query_llm_json(config, SYSTEM_PROMPT, prompt, max_tokens=8192)


def analyze_recordings(config: Config) -> Path:
    """Analyze all recording transcripts."""
    data_dir = config.data_dir
    matched_path = data_dir / "matched.json"
    out_path = data_dir / "analysis_recordings.json"

    if not matched_path.exists():
        # Fall back to schedule_clean if no matched data
        matched_path = data_dir / "schedule_clean.json"

    if not matched_path.exists():
        console.print(f"{tag('analyze')} No data found, skipping recording analysis.")
        return out_path

    sessions = json.loads(matched_path.read_text())
    sessions_with_transcripts = [s for s in sessions if s.get("transcript")]

    if not sessions_with_transcripts:
        console.print(f"{tag('analyze')} No transcripts found, skipping recording analysis.")
        return out_path

    console.print(
        f"{tag('analyze')} Analyzing {len(sessions_with_transcripts)} talk transcripts..."
    )

    # Analyze individual talks
    talk_analyses = []
    with progress_bar() as pb:
        task = pb.add_task(
            f"{tag('analyze')} Analyzing talks", total=len(sessions_with_transcripts)
        )
        for session in sessions_with_transcripts:
            title = session["title"][:50]
            try:
                result = analyze_single_talk(config, session)
                if result:
                    talk_analyses.append(result)
                pb.update(task, advance=1, description=f"{tag('analyze')} {title}")
            except Exception as e:
                pb.update(
                    task, advance=1,
                    description=f"{tag('analyze')} {title} [red]failed[/red]",
                )
                console.print(f"  {tag('analyze')} [red]{title} — {e}[/red]")

    # Save individual analyses
    individual_path = data_dir / "analysis_talks.json"
    individual_path.write_text(json.dumps(talk_analyses, indent=2, ensure_ascii=False))

    # Synthesize across talks
    if talk_analyses:
        console.print(
            f"{tag('analyze')} Synthesizing insights across {len(talk_analyses)} talks..."
        )
        with console.status(f"{tag('analyze')} Synthesizing with Claude..."):
            synthesis = synthesize_analyses(config, talk_analyses)
        synthesis["individual_talks"] = talk_analyses
    else:
        synthesis = {"individual_talks": [], "narrative": "No talks analyzed."}

    out_path.write_text(json.dumps(synthesis, indent=2, ensure_ascii=False))
    console.print(f"{tag('analyze')} Recording analysis saved to {out_path}")
    return out_path
