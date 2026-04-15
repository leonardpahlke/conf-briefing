"""Recording analysis: summarize transcripts, extract insights."""

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from conf_briefing.analyze.llm import query_llm_json
from conf_briefing.analyze.schemas import (
    SynthActions,
    SynthNarrative,
    SynthSignals,
    SynthTensions,
    TalkCore,
    TalkSignals,
)
from conf_briefing.config import MIN_VIDEO_DURATION_SEC, Config
from conf_briefing.console import console, progress_bar, tag
from conf_briefing.io import load_json_file

# --- Magic numbers ---
_MAX_TRANSCRIPT_CHARS = 30_000
_MIN_TRANSCRIPT_CHARS = 100


def _filter_empty_relationships(items: list[dict], key: str = "entity_a") -> list[dict]:
    """Remove relationship entries where the given key is empty or whitespace."""
    return [r for r in items if r.get(key, "").strip()]


SYSTEM_PROMPT = """\
You are a conference talk analyst. Extract key insights from transcripts as JSON. \
Only include information explicitly stated. If a field has no evidence, leave it empty."""

# --- Patterns for non-analytical sessions (compiled once) ---
_SKIP_PATTERNS = [
    re.compile(r"\bclosing\s+remarks?\b", re.IGNORECASE),
    re.compile(r"\bawards?\s+ceremon", re.IGNORECASE),
    re.compile(r"\bwelcome\b.*\bopening\s+remarks?\b", re.IGNORECASE),
    re.compile(r"\bopening\s+keynote\s+remarks?\b", re.IGNORECASE),
    re.compile(r"\bsponsored?\s+keynote\b", re.IGNORECASE),
]


def _is_non_analytical(title: str) -> bool:
    """Return True if the session title matches a non-analytical pattern."""
    return any(p.search(title) for p in _SKIP_PATTERNS)


# --- Talk analysis: split into 2 focused calls ---

TALK_CORE_PROMPT = """\
Analyze this conference talk transcript and extract the core content.

Title: {title}
Speakers: {speakers}
Track: {track}

Transcript:
{transcript}

Extract as JSON:
- title: exact talk title
- summary: 3-5 sentences capturing the core message. Include any concrete metrics stated.
- key_takeaways: 3-5 most important points an attendee should remember. Be specific and actionable.
- problems_discussed: 2-4 technical challenges the speaker addresses. Name the specific problem.
- tools_and_projects: all technologies, tools, OSS projects mentioned by name.
- qa_highlights: notable Q&A exchanges, if present in transcript. Empty list if no Q&A.
- evidence_quality: "production" (real workload data), "proof_of_concept" (demo/prototype), \
"theoretical" (design/proposal), "vendor_demo" (vendor showing product).
- speaker_perspective: "practitioner" (building/operating), "vendor" (selling), \
"maintainer" (OSS maintainer), "academic" (researcher). Judge by content, not job title.
- references: URLs, GitHub repos, paper citations from slides or transcript. Empty list if none.

IMPORTANT: key_takeaways, problems_discussed, and tools_and_projects must each have at least 1 item."""

TALK_SIGNALS_PROMPT = """\
Analyze this conference talk transcript for technology signals and relationships.

Title: {title}
Speakers: {speakers}
Track: {track}

Transcript:
{transcript}

Extract as JSON:
- maturity_assessments: per-technology assessment. For each technology mentioned, provide:
  - technology: name (e.g. "Kueue", "Cilium")
  - maturity: "assess" (worth looking at), "trial" (safe to experiment), "adopt" (proven), "hold" (reconsider)
  - evidence: brief evidence from the talk
  List at least 1 technology.
- caveats_and_concerns: limitations, warnings, or open problems mentioned. List at least 1.
- technology_stance: per-technology sentiment:
  - technology: name
  - stance: "enthusiastic" (strong advocacy), "cautious" (mentions risks), "critical" (argues against), "neutral"
  - evidence: brief quote or paraphrase
- relationships: technology relationships explicitly stated. BOTH entity_a and entity_b must be \
named technologies, e.g. "Kueue competes_with Volcano", "Cilium replaces kube-proxy".
  - entity_a: first technology name (MUST NOT be empty)
  - relation: one of "replaces", "competes_with", "builds_on", "integrates_with", "extends"
  - entity_b: second technology name"""


def analyze_single_talk(config: Config, session: dict) -> dict | None:
    """Analyze a single talk transcript via 2 focused LLM calls."""
    transcript = session.get("transcript", "")
    if not transcript or len(transcript) < _MIN_TRANSCRIPT_CHARS:
        return None

    # Skip short videos (highlight reels, teasers)
    duration = session.get("duration_sec", 0)
    if duration and duration < MIN_VIDEO_DURATION_SEC:
        return None

    # Skip non-analytical sessions
    if _is_non_analytical(session.get("title", "")):
        return None

    # Prefer aligned slide-transcript content, fall back to flat slide text
    slide_content = session.get("slide_aligned", "") or session.get("slide_text", "")
    if slide_content:
        label = "SLIDE-ALIGNED CONTENT" if session.get("slide_aligned") else "SLIDE CONTENT"
        transcript = transcript + f"\n\n[{label}]\n{slide_content}"

    # Append slide references if available
    refs = session.get("slide_references", [])
    if refs:
        transcript = transcript + "\n\n[SLIDE REFERENCES]\n" + "\n".join(refs)

    speakers = (
        ", ".join(
            f"{s['name']} ({s['company']})" if s.get("company") else s["name"]
            for s in session.get("speakers", [])
        )
        or "Unknown"
    )

    fmt_kwargs = {
        "title": session["title"],
        "speakers": speakers,
        "track": session.get("track", ""),
        "transcript": transcript[:_MAX_TRANSCRIPT_CHARS],
    }

    # Fire both LLM calls in parallel (independent inputs)
    with ThreadPoolExecutor(max_workers=2) as pool:
        core_future = pool.submit(
            query_llm_json,
            config,
            SYSTEM_PROMPT,
            TALK_CORE_PROMPT.format(**fmt_kwargs),
            max_tokens=4096,
            schema=TalkCore,
        )
        signals_future = pool.submit(
            query_llm_json,
            config,
            SYSTEM_PROMPT,
            TALK_SIGNALS_PROMPT.format(**fmt_kwargs),
            max_tokens=4096,
            schema=TalkSignals,
        )
        core_result = core_future.result()
        signals_result = signals_future.result()

    # Merge into single dict
    merged = {**core_result, **signals_result}

    # Post-processing: drop relationships where entity_a is empty
    if "relationships" in merged:
        merged["relationships"] = _filter_empty_relationships(merged["relationships"])

    return merged


# --- Synthesis: split into 4 focused calls ---

SYNTH_NARRATIVE_PROMPT = """\
Synthesize a narrative overview from these conference talk analyses.

Conference: "{conference_name}"

Individual talk analyses:
{analyses_json}

Produce JSON with:
- narrative: 4-6 paragraph markdown overview of key insights across all talks. \
Cover the major themes, notable trends, and overall conference direction.
- cross_cutting_themes: themes appearing across multiple talks. Identify at least 4 themes. Each with:
  - theme: short name
  - description: 2-3 sentences
  - supporting_talks: list of talk titles that discuss this theme
- common_problems: shared technical challenges across talks. List at least 5 specific problems."""

SYNTH_SIGNALS_PROMPT = """\
Extract technology signals from these conference talk analyses.

Conference: "{conference_name}"

Individual talk analyses:
{analyses_json}

Produce JSON with:
- emerging_technologies: technologies mentioned across multiple talks. For each:
  - technology: name
  - mentions: count of talks mentioning it
  - context: 1-2 sentence summary of how it was discussed
  List at least 5 technologies.
- technology_relationships: aggregate technology relationships from individual talks. \
BOTH entity_a and entity_b must be named technologies. Merge duplicates, list supporting talks.
  - entity_a: first technology (MUST NOT be empty)
  - relation: relationship type
  - entity_b: second technology
  - supporting_talks: talk titles"""

SYNTH_TENSIONS_PROMPT = """\
Identify tensions and maturity assessments from these conference talk analyses.

Conference: "{conference_name}"

Individual talk analyses:
{analyses_json}

Produce JSON with:
- tensions: talks that contradict each other or present opposing approaches. For each:
  - topic: what the disagreement is about
  - side_a: {{position, supporting_talks}}
  - side_b: {{position, supporting_talks}}
  - severity: "fundamental", "significant", or "minor"
  - implication: what this means for practitioners
  Identify at least 2 tensions.
- maturity_landscape: aggregate maturity assessments. One entry per technology. For each:
  - technology: name
  - ring: "assess", "trial", "adopt", or "hold"
  - evidence_quality: "anecdotal", "case_study", "benchmarked", or "production_proven"
  - supporting_talks: talk titles
  - rationale: why this ring placement
  List at least 5 technologies."""

SYNTH_ACTIONS_PROMPT = """\
Recommend actions based on these conference talk analyses.

Conference: "{conference_name}"

Individual talk analyses:
{analyses_json}

Produce JSON with:
- recommended_actions: concrete next steps for an attendee. Be specific and actionable. For each:
  - action: specific thing to do (e.g. "Evaluate Kueue for batch ML workloads")
  - category: "evaluate", "watch", "talk_to", "adopt", or "avoid"
  - urgency: "immediate", "next_quarter", or "long_term"
  - supporting_evidence: brief evidence from the talks
  List at least 5 actions."""


def _condense_for_synthesis(analyses: list[dict]) -> list[dict]:
    """Trim per-talk analyses to essential fields for synthesis prompt."""
    keep_fields = [
        "title",
        "summary",
        "key_takeaways",
        "tools_and_projects",
        "problems_discussed",
        "evidence_quality",
        "speaker_perspective",
        "maturity_assessments",
        "caveats_and_concerns",
        "technology_stance",
        "relationships",
    ]
    condensed = []
    for talk in analyses:
        entry = {k: talk[k] for k in keep_fields if k in talk}
        condensed.append(entry)
    return condensed


def synthesize_analyses(config: Config, analyses: list[dict]) -> dict:
    """Synthesize insights across all talk analyses via 4 focused LLM calls."""
    condensed = _condense_for_synthesis(analyses)
    analyses_json = json.dumps(condensed, indent=2, ensure_ascii=False)

    fmt_kwargs = {
        "conference_name": config.conference.name,
        "analyses_json": analyses_json,
    }

    # Fire all 4 synthesis calls in parallel (independent inputs)
    with ThreadPoolExecutor(max_workers=4) as pool:
        narrative_future = pool.submit(
            query_llm_json,
            config,
            SYSTEM_PROMPT,
            SYNTH_NARRATIVE_PROMPT.format(**fmt_kwargs),
            max_tokens=8192,
            schema=SynthNarrative,
        )
        signals_future = pool.submit(
            query_llm_json,
            config,
            SYSTEM_PROMPT,
            SYNTH_SIGNALS_PROMPT.format(**fmt_kwargs),
            max_tokens=4096,
            schema=SynthSignals,
        )
        tensions_future = pool.submit(
            query_llm_json,
            config,
            SYSTEM_PROMPT,
            SYNTH_TENSIONS_PROMPT.format(**fmt_kwargs),
            max_tokens=6144,
            schema=SynthTensions,
        )
        actions_future = pool.submit(
            query_llm_json,
            config,
            SYSTEM_PROMPT,
            SYNTH_ACTIONS_PROMPT.format(**fmt_kwargs),
            max_tokens=4096,
            schema=SynthActions,
        )
        narrative_result = narrative_future.result()
        signals_result = signals_future.result()
        tensions_result = tensions_future.result()
        actions_result = actions_future.result()

    # Merge all results
    merged = {**narrative_result, **signals_result, **tensions_result, **actions_result}

    # Post-processing: drop technology_relationships where entity_a is empty
    if "technology_relationships" in merged:
        merged["technology_relationships"] = _filter_empty_relationships(
            merged["technology_relationships"]
        )

    return merged


def analyze_recordings(config: Config) -> Path | None:
    """Analyze all recording transcripts."""
    data_dir = config.data_dir
    matched_path = data_dir / "matched.json"
    out_path = data_dir / "analysis_recordings.json"

    if not matched_path.exists():
        # Fall back to schedule_clean if no matched data
        matched_path = data_dir / "schedule_clean.json"

    if not matched_path.exists():
        console.print(f"{tag('analyze')} No data found, skipping recording analysis.")
        return None

    sessions = load_json_file(matched_path)
    sessions_with_transcripts = [s for s in sessions if s.get("transcript")]

    if not sessions_with_transcripts:
        console.print(f"{tag('analyze')} No transcripts found, skipping recording analysis.")
        return None

    # Filter non-analytical sessions
    skipped = [s for s in sessions_with_transcripts if _is_non_analytical(s.get("title", ""))]
    sessions_with_transcripts = [
        s for s in sessions_with_transcripts if not _is_non_analytical(s.get("title", ""))
    ]
    if skipped:
        console.print(
            f"{tag('analyze')} Skipping {len(skipped)} non-analytical sessions "
            f"({', '.join(s['title'][:40] for s in skipped[:3])}{'...' if len(skipped) > 3 else ''})"
        )

    # Load existing analyses to skip already-analyzed talks
    individual_path = data_dir / "analysis_talks.json"
    existing_analyses: list[dict] = []
    already_analyzed: set[str] = set()
    if individual_path.exists():
        existing_analyses = load_json_file(individual_path)
        already_analyzed = {t["title"] for t in existing_analyses if t.get("title")}

    to_analyze = [s for s in sessions_with_transcripts if s["title"] not in already_analyzed]
    skipped_existing = len(sessions_with_transcripts) - len(to_analyze)

    if skipped_existing:
        console.print(
            f"{tag('analyze')} Skipping {skipped_existing} already-analyzed talk(s)."
        )

    if not to_analyze:
        console.print(f"{tag('analyze')} All talks already analyzed.")
        # Still re-synthesize if synthesis file is missing
        if not out_path.exists() and existing_analyses:
            console.print(
                f"{tag('analyze')} Synthesizing insights across {len(existing_analyses)} talks..."
            )
            with console.status(f"{tag('analyze')} Synthesizing..."):
                synthesis = synthesize_analyses(config, existing_analyses)
            synthesis["individual_talks"] = existing_analyses
            out_path.write_text(json.dumps(synthesis, indent=2, ensure_ascii=False))
            console.print(f"{tag('analyze')} Recording analysis saved to {out_path}")
        return out_path

    console.print(
        f"{tag('analyze')} Analyzing {len(to_analyze)} talk transcripts..."
    )

    # Analyze individual talks (parallel), saving incrementally
    new_analyses = []
    failed_count = 0

    def _save_progress():
        """Persist current results so interrupted runs can resume."""
        from conf_briefing.analyze.entities import canonicalize_analysis as _canon

        all_so_far = existing_analyses + [_canon(t) for t in new_analyses]
        individual_path.write_text(json.dumps(all_so_far, indent=2, ensure_ascii=False))

    with progress_bar() as pb:
        task = pb.add_task(
            f"{tag('analyze')} Analyzing talks", total=len(to_analyze)
        )

        def _analyze(session):
            return session["title"][:50], analyze_single_talk(config, session)

        with ThreadPoolExecutor(max_workers=config.llm.num_parallel) as executor:
            futures = {executor.submit(_analyze, s): s for s in to_analyze}
            for future in as_completed(futures):
                try:
                    title, result = future.result()
                    if result:
                        new_analyses.append(result)
                        _save_progress()
                    pb.update(task, advance=1, description=f"{tag('analyze')} {title}")
                except Exception as e:
                    failed_count += 1
                    session = futures[future]
                    title = session["title"][:50]
                    pb.update(
                        task,
                        advance=1,
                        description=f"{tag('analyze')} {title} [red]failed[/red]",
                    )
                    console.print(f"  {tag('analyze')} [red]{title} — {e}[/red]")

    if failed_count:
        total_attempted = len(to_analyze)
        console.print(
            f"{tag('analyze')} [yellow]{failed_count}/{total_attempted} talks failed analysis[/yellow]"
        )
        if failed_count > total_attempted / 2:
            console.print(
                f"{tag('analyze')} [red bold]Warning: >50% of talks failed. "
                f"Check Ollama connectivity and model availability.[/red bold]"
            )

    # Apply entity canonicalization to new results
    from conf_briefing.analyze.entities import canonicalize_analysis

    new_analyses = [canonicalize_analysis(t) for t in new_analyses]

    # Merge with existing analyses and do a final save
    talk_analyses = existing_analyses + new_analyses
    individual_path.write_text(json.dumps(talk_analyses, indent=2, ensure_ascii=False))
    console.print(
        f"{tag('analyze')} Saved {len(new_analyses)} new + {len(existing_analyses)} existing = {len(talk_analyses)} total analyses."
    )

    # Synthesize across all talks (existing + new)
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
