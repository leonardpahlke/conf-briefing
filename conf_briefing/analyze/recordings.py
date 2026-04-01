"""Recording analysis: summarize transcripts, extract insights."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from conf_briefing.analyze.llm import query_llm_json
from conf_briefing.analyze.schemas import SynthesisResult, TalkAnalysis
from conf_briefing.config import MIN_VIDEO_DURATION_SEC, Config
from conf_briefing.console import console, progress_bar, tag

SYSTEM_PROMPT = """\
You are a conference talk analyst. Extract key insights from transcripts as JSON. \
Only include information explicitly stated. If a field has no evidence, leave it empty."""

SINGLE_TALK_PROMPT = """\
Analyze this conference talk transcript.

Title: {title}
Speakers: {speakers}
Track: {track}

Transcript:
{transcript}

Extract (JSON schema enforced — focus on content quality):
- title: exact talk title.
- summary: 3-5 sentences capturing the core message. Include any concrete metrics stated.
- key_takeaways: most important points an attendee should remember.
- problems_discussed: technical challenges the speaker addresses.
- tools_and_projects: technologies, tools, OSS projects mentioned.
- qa_highlights: notable Q&A exchanges, if present in transcript.
- evidence_quality: "production" (real workload data), "proof_of_concept" (demo/prototype), \
"theoretical" (design/proposal), "vendor_demo" (vendor showing product).
- speaker_perspective: "practitioner" (building/operating), "vendor" (selling), \
"maintainer" (OSS maintainer), "academic" (researcher). Judge by content, not job title.
- maturity_assessments: per-technology. "assess" (worth looking at), "trial" (safe to experiment), \
"adopt" (proven), "hold" (reconsider). Include brief evidence.
- caveats_and_concerns: limitations, warnings, or open problems mentioned.
- technology_stance: per-technology sentiment. "enthusiastic" (strong advocacy), \
"cautious" (mentions risks), "critical" (argues against), "neutral" (informational).
- relationships: technology relationships explicitly stated, e.g. Cilium replaces kube-proxy.
- references: URLs, GitHub repos, paper citations from slides or transcript."""

SYNTHESIS_PROMPT = """\
Synthesize insights across these conference talk analyses.

Conference: "{conference_name}"

Individual talk analyses:
{analyses_json}

Produce (JSON schema enforced — focus on content quality):
- narrative: 4-6 paragraph markdown overview of key insights across all talks.
- cross_cutting_themes: themes appearing across multiple talks, with supporting talk titles.
- emerging_technologies: technologies mentioned across talks, with mention count and context.
- common_problems: shared challenges across talks.
- tensions: talks that contradict each other or present opposing approaches. \
Include both sides, severity (fundamental/significant/minor), and practitioner implications.
- maturity_landscape: aggregate maturity assessments from individual talks. \
One entry per technology. Include evidence quality and rationale.
- recommended_actions: concrete next steps for an attendee. Be specific and actionable.
- technology_relationships: aggregate from individual talks. \
Merge duplicates, list supporting talks."""


def analyze_single_talk(config: Config, session: dict) -> dict | None:
    """Analyze a single talk transcript."""
    transcript = session.get("transcript", "")
    if not transcript or len(transcript) < 100:
        return None

    # Skip short videos (highlight reels, teasers)
    duration = session.get("duration_sec", 0)
    if duration and duration < MIN_VIDEO_DURATION_SEC:
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

    prompt = SINGLE_TALK_PROMPT.format(
        title=session["title"],
        speakers=speakers,
        track=session.get("track", ""),
        transcript=transcript[:30000],  # Cap at ~30K chars (qwen3:32b supports 32K+ context)
    )

    result = query_llm_json(
        config, SYSTEM_PROMPT, prompt, max_tokens=6144, schema=TalkAnalysis
    )
    return result


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
    """Synthesize insights across all talk analyses."""
    condensed = _condense_for_synthesis(analyses)
    prompt = SYNTHESIS_PROMPT.format(
        conference_name=config.conference.name,
        analyses_json=json.dumps(condensed, indent=2, ensure_ascii=False),
    )

    result = query_llm_json(
        config, SYSTEM_PROMPT, prompt, max_tokens=12288, schema=SynthesisResult
    )
    return result


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

        with ThreadPoolExecutor(max_workers=config.llm.num_parallel) as executor:
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

    # Apply entity canonicalization
    from conf_briefing.analyze.entities import canonicalize_analysis

    talk_analyses = [canonicalize_analysis(t) for t in talk_analyses]

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
