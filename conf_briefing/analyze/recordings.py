"""Recording analysis: summarize transcripts, extract insights."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
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
  "summary": "3-5 sentence summary",
  "evidence_quality": "production | proof_of_concept | theoretical | vendor_demo",
  "speaker_perspective": "practitioner | vendor | maintainer | academic",
  "maturity_assessments": [
    {{"technology": "name", "maturity": "assess | trial | adopt | hold", "evidence": "brief evidence"}}
  ],
  "concrete_metrics": ["40% latency reduction", "5x deployment speed"],
  "caveats_and_concerns": ["scaling issues above 10k nodes"],
  "audience_energy": "high | moderate | low"
}}

Notes:
- evidence_quality: "production" = real workload data; "proof_of_concept" = demo or prototype; \
"theoretical" = design/proposal; "vendor_demo" = vendor showing their product.
- speaker_perspective: judge by content, not job title. Vendors presenting case studies = "practitioner".
- maturity: "assess" = worth looking at; "trial" = safe to experiment; "adopt" = proven; "hold" = reconsider.
- concrete_metrics: only include numbers explicitly stated in the talk. Leave empty if none.
- caveats_and_concerns: limitations, warnings, or open problems the speaker mentioned.
- audience_energy: based on Q&A depth, applause cues, or engagement signals in transcript."""

SYNTHESIS_PROMPT = """\
Synthesize insights across these conference talk analyses.

Conference: "{conference_name}"
Focus: {focus_topics}

Individual talk analyses:
{analyses_json}

Produce a JSON object with the following structure. Include all top-level keys even if \
their arrays are empty.

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
  "narrative": "4-6 paragraph markdown synthesis of key insights across all talks.",
  "tensions": [
    {{
      "topic": "sidecar vs sidecarless",
      "side_a": {{"position": "...", "supporting_talks": ["..."]}},
      "side_b": {{"position": "...", "supporting_talks": ["..."]}},
      "severity": "fundamental | significant | minor",
      "implication": "what this means for practitioners"
    }}
  ],
  "maturity_landscape": [
    {{
      "technology": "name",
      "ring": "assess | trial | adopt | hold",
      "evidence_quality": "anecdotal | case_study | benchmarked | production_proven",
      "supporting_talks": ["..."],
      "rationale": "why this placement"
    }}
  ],
  "stakeholder_map": [
    {{
      "company": "name",
      "role": "vendor | end_user | maintainer | cloud_provider",
      "pushing": ["topics they promote"],
      "talk_count": 3,
      "notable_claims": ["claims worth scrutinizing"]
    }}
  ],
  "quiet_signals": [
    {{"signal": "...", "source_talk": "...", "why_notable": "..."}}
  ],
  "absent_topics": [
    {{"topic": "...", "expected_because": "...", "possible_reason": "..."}}
  ],
  "recommended_actions": [
    {{
      "action": "...",
      "category": "evaluate | watch | talk_to | adopt | avoid",
      "urgency": "immediate | next_quarter | long_term",
      "supporting_evidence": "..."
    }}
  ]
}}

Guidelines:
- tensions: look for talks that contradict each other or present opposing approaches.
- maturity_landscape: aggregate maturity_assessments from individual talks. One entry per technology.
- stakeholder_map: group by company. Judge role by content, not branding.
- quiet_signals: things mentioned once that could be important. Unexpected topics.
- absent_topics: topics you'd expect at this conference but nobody discussed.
- recommended_actions: concrete next steps for an attendee. Be specific."""


def analyze_single_talk(config: Config, session: dict) -> dict | None:
    """Analyze a single talk transcript."""
    transcript = session.get("transcript", "")
    if not transcript or len(transcript) < 100:
        return None

    # Append slide content if available
    slide_text = session.get("slide_text", "")
    if slide_text:
        transcript = transcript + f"\n\n[SLIDE CONTENT]\n{slide_text}"

    speakers = (
        ", ".join(
            f"{s['name']} ({s['company']})" if s.get("company") else s["name"]
            for s in session.get("speakers", [])
        )
        or "Unknown"
    )

    prompt = SINGLE_TALK_PROMPT.format(
        title=session["title"],
        speakers=speakers,
        track=session.get("track", ""),
        transcript=transcript[:15000],  # Cap at ~15K chars to stay within context
    )

    result = query_llm_json(config, SYSTEM_PROMPT, prompt, max_tokens=6144)
    return _normalize_talk_analysis(result) if isinstance(result, dict) else result


def _normalize_talk_analysis(talk: dict) -> dict:
    """Fill missing optional fields with safe defaults."""
    defaults = {
        "key_takeaways": [],
        "qa_highlights": [],
        "problems_discussed": [],
        "tools_and_projects": [],
        "emerging_signals": [],
        "summary": "",
        "evidence_quality": "theoretical",
        "speaker_perspective": "practitioner",
        "maturity_assessments": [],
        "concrete_metrics": [],
        "caveats_and_concerns": [],
        "audience_energy": "moderate",
    }
    for key, default in defaults.items():
        if key not in talk:
            talk[key] = default
    # Normalize maturity_assessments entries
    for entry in talk.get("maturity_assessments", []):
        entry.setdefault("technology", "")
        entry.setdefault("maturity", "assess")
        entry.setdefault("evidence", "")
    return talk


def _condense_for_synthesis(analyses: list[dict]) -> list[dict]:
    """Trim per-talk analyses to essential fields for synthesis prompt.

    Prevents context overflow when many talks are analyzed by keeping only
    the fields the synthesis prompt needs to reason over.
    """
    keep_fields = [
        "title",
        "summary",
        "key_takeaways",
        "tools_and_projects",
        "emerging_signals",
        "problems_discussed",
        "evidence_quality",
        "speaker_perspective",
        "maturity_assessments",
        "concrete_metrics",
        "caveats_and_concerns",
    ]
    condensed = []
    for talk in analyses:
        entry = {k: talk[k] for k in keep_fields if k in talk}
        condensed.append(entry)
    return condensed


def _normalize_synthesis(result: dict) -> dict:
    """Fill missing optional fields in synthesis output with safe defaults."""
    defaults = {
        "cross_cutting_themes": [],
        "emerging_technologies": [],
        "common_problems": [],
        "narrative": "",
        "tensions": [],
        "maturity_landscape": [],
        "stakeholder_map": [],
        "quiet_signals": [],
        "absent_topics": [],
        "recommended_actions": [],
    }
    for key, default in defaults.items():
        if key not in result:
            result[key] = default
    return result


def synthesize_analyses(config: Config, analyses: list[dict]) -> dict:
    """Synthesize insights across all talk analyses."""
    focus = ["general industry trends"]

    condensed = _condense_for_synthesis(analyses)
    prompt = SYNTHESIS_PROMPT.format(
        conference_name=config.conference.name,
        focus_topics=", ".join(focus),
        analyses_json=json.dumps(condensed, indent=2, ensure_ascii=False),
    )

    result = query_llm_json(config, SYSTEM_PROMPT, prompt, max_tokens=12288)
    return _normalize_synthesis(result) if isinstance(result, dict) else result


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

    # Analyze individual talks (parallel)
    talk_analyses = []
    with progress_bar() as pb:
        task = pb.add_task(
            f"{tag('analyze')} Analyzing talks", total=len(sessions_with_transcripts)
        )

        def _analyze(session):
            return session["title"][:50], analyze_single_talk(config, session)

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(_analyze, s): s for s in sessions_with_transcripts
            }
            for future in as_completed(futures):
                try:
                    title, result = future.result()
                    if result:
                        talk_analyses.append(result)
                    pb.update(task, advance=1, description=f"{tag('analyze')} {title}")
                except Exception as e:
                    session = futures[future]
                    title = session["title"][:50]
                    pb.update(
                        task,
                        advance=1,
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
        with console.status(f"{tag('analyze')} Synthesizing..."):
            synthesis = synthesize_analyses(config, talk_analyses)
        synthesis["individual_talks"] = talk_analyses
    else:
        synthesis = {"individual_talks": [], "narrative": "No talks analyzed."}

    out_path.write_text(json.dumps(synthesis, indent=2, ensure_ascii=False))
    console.print(f"{tag('analyze')} Recording analysis saved to {out_path}")
    return out_path
