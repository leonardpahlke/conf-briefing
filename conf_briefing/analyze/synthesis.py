"""Synthesis: per-cluster and global synthesis across talk analyses."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from conf_briefing.analyze.llm import query_llm_json
from conf_briefing.analyze.schemas import (
    SynthActions,
    SynthNarrative,
    SynthSignals,
    SynthTensions,
)
from conf_briefing.config import Config
from conf_briefing.console import console, tag
from conf_briefing.io import load_json_file

# --- Shared helpers ---


def _filter_empty_relationships(items: list[dict], key: str = "entity_a") -> list[dict]:
    """Remove relationship entries where the given key is empty or whitespace."""
    return [r for r in items if r.get(key, "").strip()]


SYNTH_SYSTEM_PROMPT = """\
You are a conference analyst. Synthesize insights across talk analyses within a topic cluster. \
Only include claims supported by the analyses provided."""

GLOBAL_SYSTEM_PROMPT = """\
You are a conference analyst. Synthesize insights across cluster summaries to identify \
conference-wide patterns. Only include claims supported by the cluster summaries provided."""

# --- Per-cluster synthesis prompts ---

SYNTH_NARRATIVE_PROMPT = """\
Synthesize a narrative overview from these conference talk analyses within a topic cluster.

Conference: "{conference_name}"

Individual talk analyses:
{analyses_json}

Produce JSON with:
- narrative: 4-6 paragraph markdown overview of key insights across these talks. \
Cover the major themes, notable trends, and the overall direction of this topic area. \
Weight production evidence more heavily than theoretical or vendor_demo evidence.
- cross_cutting_themes: key insights — distinct angles or findings within this cluster. \
Identify 2-5 key insights. Each with:
  - theme: short name
  - description: 2-3 sentences
  - supporting_talks: list of talk titles that discuss this theme
- common_problems: key technical challenges discussed (at least 2)."""

SYNTH_SIGNALS_PROMPT = """\
Extract technology signals from these conference talk analyses within a topic cluster.

Conference: "{conference_name}"

Individual talk analyses:
{analyses_json}

Produce JSON with:
- emerging_technologies: the most prominent technologies that appear in these talks. For each:
  - technology: name
  - mentions: estimated count of talks mentioning it
  - context: 1-2 sentence summary of how it was discussed
  List at least 3 technologies.
- technology_relationships: aggregate technology relationships from individual talks. \
BOTH entity_a and entity_b must be named technologies. Merge duplicates, list supporting talks.
  - entity_a: first technology (MUST NOT be empty)
  - relation: relationship type
  - entity_b: second technology
  - supporting_talks: talk titles"""

SYNTH_TENSIONS_PROMPT = """\
Identify tensions and maturity assessments from these conference talk analyses within a topic cluster.

Conference: "{conference_name}"

Individual talk analyses:
{analyses_json}

Produce JSON with:
- tensions: identify tensions where talks present genuinely opposing approaches. \
If no real tensions exist, return an empty list. For each:
  - topic: what the disagreement is about
  - side_a: {{position, supporting_talks}}
  - side_b: {{position, supporting_talks}}
  - severity: "fundamental", "significant", or "minor"
  - implication: what this means for practitioners
- maturity_landscape: aggregate maturity assessments. One entry per technology. For each:
  - technology: name
  - ring: "assess", "trial", "adopt", or "hold"
  - evidence_quality: "anecdotal", "case_study", "benchmarked", or "production_proven"
  - supporting_talks: talk titles
  - rationale: why this ring placement
  List at least 3 technologies."""

SYNTH_ACTIONS_PROMPT = """\
Recommend actions based on these conference talk analyses within a topic cluster.

Conference: "{conference_name}"

Individual talk analyses:
{analyses_json}

Produce JSON with:
- recommended_actions: concrete next steps for a cloud-native practitioner \
(platform engineer, SRE, or DevOps engineer). Be specific and actionable. \
Prioritize recommendations backed by production evidence over theoretical or vendor claims. For each:
  - action: specific thing to do (e.g. "Evaluate Kueue for batch ML workloads")
  - category: "evaluate", "watch", "talk_to", "adopt", or "avoid"
  - urgency: "immediate", "next_quarter", or "long_term"
  - supporting_evidence: 1-2 sentences of evidence from the talks
  List at least 3 actions."""

# --- Global synthesis prompts (synthesize ACROSS cluster summaries) ---

GLOBAL_NARRATIVE_PROMPT = """\
Synthesize a narrative overview ACROSS all cluster summaries from this conference.

Conference: "{conference_name}"

Cluster summaries:
{analyses_json}

Produce JSON with:
- narrative: 4-6 paragraph markdown overview of the ENTIRE conference. \
Identify how different clusters connect, what overarching direction the conference reveals, \
and what the most important cross-cluster insights are. \
Weight production evidence more heavily than theoretical or vendor_demo evidence.
- cross_cutting_themes: themes that span MULTIPLE clusters. Identify 2-5 themes. Each with:
  - theme: short name
  - description: 2-3 sentences explaining how this theme manifests across clusters
  - supporting_talks: list the cluster names (not individual talk titles) that support each theme
- common_problems: shared technical challenges that appear across clusters. List at least 3."""

GLOBAL_SIGNALS_PROMPT = """\
Extract technology signals ACROSS all cluster summaries from this conference.

Conference: "{conference_name}"

Cluster summaries:
{analyses_json}

Produce JSON with:
- emerging_technologies: technologies mentioned across multiple clusters. For each:
  - technology: name
  - mentions: count of clusters where it appears
  - context: 1-2 sentence summary of how it was discussed across clusters
  List at least 3 technologies.
- technology_relationships: aggregate relationships from cluster summaries. \
BOTH entity_a and entity_b must be named technologies. Merge duplicates.
  - entity_a: first technology (MUST NOT be empty)
  - relation: relationship type
  - entity_b: second technology
  - supporting_talks: list the cluster names (not individual talk titles) that evidence this relationship"""

GLOBAL_TENSIONS_PROMPT = """\
Identify tensions and maturity assessments ACROSS all cluster summaries from this conference.

Conference: "{conference_name}"

Cluster summaries:
{analyses_json}

Produce JSON with:
- tensions: contradictions or opposing approaches that emerge ACROSS clusters. \
If no genuine cross-cluster tensions exist, return an empty list. For each:
  - topic: what the disagreement is about
  - side_a: {{position, supporting_talks}} — use cluster names for supporting_talks
  - side_b: {{position, supporting_talks}} — use cluster names for supporting_talks
  - severity: "fundamental", "significant", or "minor"
  - implication: what this means for practitioners
  Identify at least 1 tension if genuine disagreement exists.
- maturity_landscape: aggregate maturity assessments across all clusters. One entry per technology:
  - technology: name
  - ring: "assess", "trial", "adopt", or "hold"
  - evidence_quality: "anecdotal", "case_study", "benchmarked", or "production_proven"
  - supporting_talks: list the cluster names (not individual talk titles) that assess this technology
  - rationale: why this ring placement
  List at least 3 technologies."""

GLOBAL_ACTIONS_PROMPT = """\
Recommend actions based on insights ACROSS all cluster summaries from this conference.

Conference: "{conference_name}"

Cluster summaries:
{analyses_json}

Produce JSON with:
- recommended_actions: concrete next steps for a cloud-native practitioner \
(platform engineer, SRE, or DevOps engineer) synthesized across ALL clusters. \
Be specific and actionable. \
Prioritize recommendations backed by production evidence over theoretical or vendor claims. \
If multiple clusters suggest similar actions, consolidate them into a single recommendation \
citing evidence from all relevant clusters. For each:
  - action: specific thing to do (e.g. "Evaluate Kueue for batch ML workloads")
  - category: "evaluate", "watch", "talk_to", "adopt", or "avoid"
  - urgency: "immediate", "next_quarter", or "long_term"
  - supporting_evidence: 1-2 sentences of evidence from across clusters
  List at least 3 actions."""


# --- Shared condensation ---


def _condense_for_synthesis(analyses: list[dict]) -> list[dict]:
    """Trim per-talk analyses to essential fields for synthesis prompt."""
    keep_fields = [
        "title",
        "speakers",
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


def _condense_for_global(cluster_syntheses: list[dict]) -> list[dict]:
    """Trim cluster syntheses to essential fields for the global prompt."""
    keep_fields = [
        "cluster_name",
        "narrative",
        "cross_cutting_themes",
        "common_problems",
        "emerging_technologies",
        "technology_relationships",
        "tensions",
        "maturity_landscape",
        "recommended_actions",
    ]
    condensed = []
    for cs in cluster_syntheses:
        entry = {k: cs[k] for k in keep_fields if k in cs}
        condensed.append(entry)
    return condensed


# --- Core synthesis function (4 parallel LLM calls) ---


def synthesize_analyses(
    config: Config,
    analyses: list[dict],
    *,
    system_prompt: str = SYNTH_SYSTEM_PROMPT,
    narrative_prompt: str = SYNTH_NARRATIVE_PROMPT,
    signals_prompt: str = SYNTH_SIGNALS_PROMPT,
    tensions_prompt: str = SYNTH_TENSIONS_PROMPT,
    actions_prompt: str = SYNTH_ACTIONS_PROMPT,
    condense: bool = True,
) -> dict:
    """Synthesize insights via 4 focused LLM calls.

    When condense=True, trims per-talk analyses to essential fields first.
    Pass custom prompts for global synthesis.
    """
    data = _condense_for_synthesis(analyses) if condense else analyses
    analyses_json = json.dumps(data, indent=2, ensure_ascii=False)

    fmt_kwargs = {
        "conference_name": config.conference.name,
        "analyses_json": analyses_json,
    }

    with ThreadPoolExecutor(max_workers=4) as pool:
        narrative_future = pool.submit(
            query_llm_json,
            config,
            system_prompt,
            narrative_prompt.format(**fmt_kwargs),
            max_tokens=8192,
            schema=SynthNarrative,
        )
        signals_future = pool.submit(
            query_llm_json,
            config,
            system_prompt,
            signals_prompt.format(**fmt_kwargs),
            max_tokens=4096,
            schema=SynthSignals,
        )
        tensions_future = pool.submit(
            query_llm_json,
            config,
            system_prompt,
            tensions_prompt.format(**fmt_kwargs),
            max_tokens=6144,
            schema=SynthTensions,
        )
        actions_future = pool.submit(
            query_llm_json,
            config,
            system_prompt,
            actions_prompt.format(**fmt_kwargs),
            max_tokens=4096,
            schema=SynthActions,
        )
        narrative_result = narrative_future.result()
        signals_result = signals_future.result()
        tensions_result = tensions_future.result()
        actions_result = actions_future.result()

    merged = {**narrative_result, **signals_result, **tensions_result, **actions_result}

    if "technology_relationships" in merged:
        merged["technology_relationships"] = _filter_empty_relationships(
            merged["technology_relationships"]
        )

    return merged


# --- Phase 4: Per-cluster synthesis ---


def synthesize_clusters(config: Config) -> Path | None:
    """Synthesize insights per cluster (Phase 4).

    Iterates clusters sequentially, running 4 parallel LLM calls per cluster.
    Incremental: skips clusters already present in analysis_clusters.json.
    """
    data_dir = config.data_dir
    agenda_path = data_dir / "analysis_agenda.json"
    talks_path = data_dir / "analysis_talks.json"
    out_path = data_dir / "analysis_clusters.json"

    if not agenda_path.exists() or not talks_path.exists():
        console.print(
            f"{tag('analyze')} Missing agenda or talks analysis, skipping cluster synthesis."
        )
        return None

    agenda = load_json_file(agenda_path)
    clusters = agenda.get("clusters", [])
    if not clusters:
        console.print(f"{tag('analyze')} No clusters found, skipping cluster synthesis.")
        return None

    all_talks = load_json_file(talks_path)
    talks_by_title: dict[str, dict] = {t["title"]: t for t in all_talks if t.get("title")}

    # Build prefix lookup for fuzzy matching: cluster titles from the LLM
    # may omit the " - Speaker, Org" suffix present in analysis_talks.json.
    talks_by_prefix: dict[str, dict] = {}
    for title, talk in talks_by_title.items():
        # Key by the part before " - " (speaker separator)
        prefix = title.split(" - ")[0].strip()
        if prefix not in talks_by_prefix:
            talks_by_prefix[prefix] = talk

    def _resolve_talk(title: str) -> dict | None:
        """Resolve a cluster talk title to its full analysis, with prefix fallback."""
        if title in talks_by_title:
            return talks_by_title[title]
        # Fallback: match by prefix (title without speaker suffix)
        prefix = title.split(" - ")[0].strip()
        if prefix in talks_by_prefix:
            return talks_by_prefix[prefix]
        # Fallback: check if any full title starts with the cluster title
        for full_title, talk in talks_by_title.items():
            if full_title.startswith(title):
                return talk
        return None

    # Load existing cluster syntheses for incremental processing
    existing: list[dict] = []
    already_done: set[str] = set()
    if out_path.exists():
        existing = load_json_file(out_path)
        already_done = {cs["cluster_name"] for cs in existing if cs.get("cluster_name")}

    new_syntheses: list[dict] = []
    skipped = 0

    console.print(
        f"{tag('analyze')} Synthesizing {len(clusters)} clusters..."
    )

    for i, cluster in enumerate(clusters, 1):
        name = cluster.get("name", f"cluster_{i}")

        if name in already_done:
            skipped += 1
            continue

        # Gather talks for this cluster
        cluster_talk_list = [
            resolved
            for title in cluster.get("talks", [])
            if (resolved := _resolve_talk(title)) is not None
        ]

        if not cluster_talk_list:
            console.print(
                f"  {tag('analyze')} Cluster '{name}' — no matching talks, skipping."
            )
            continue

        console.print(
            f"  {tag('analyze')} [{i}/{len(clusters)}] Synthesizing '{name}' "
            f"({len(cluster_talk_list)} talks)..."
        )

        synthesis = synthesize_analyses(config, cluster_talk_list)
        synthesis["cluster_name"] = name
        new_syntheses.append(synthesis)

        # Save incrementally
        all_so_far = existing + new_syntheses
        out_path.write_text(json.dumps(all_so_far, indent=2, ensure_ascii=False))

    if skipped:
        console.print(
            f"{tag('analyze')} Skipped {skipped} already-synthesized cluster(s)."
        )

    total = len(existing) + len(new_syntheses)
    console.print(
        f"{tag('analyze')} Cluster synthesis complete: {total} clusters saved to {out_path}"
    )
    return out_path


# --- Phase 5: Global synthesis ---


def synthesize_global(config: Config) -> Path | None:
    """Synthesize insights across all cluster summaries (Phase 5).

    Reads analysis_clusters.json, condenses, and runs 4 parallel LLM calls
    with global prompts that instruct cross-cluster synthesis.
    Skips if analysis_recordings.json is newer than analysis_clusters.json.
    """
    data_dir = config.data_dir
    clusters_path = data_dir / "analysis_clusters.json"
    out_path = data_dir / "analysis_recordings.json"

    if not clusters_path.exists():
        console.print(
            f"{tag('analyze')} No cluster synthesis found, skipping global synthesis."
        )
        return None

    # Cache check: skip if output is newer than input
    if out_path.exists() and out_path.stat().st_mtime > clusters_path.stat().st_mtime:
        console.print(f"{tag('analyze')} Global synthesis is up-to-date, skipping.")
        return out_path

    cluster_syntheses = load_json_file(clusters_path)
    if not cluster_syntheses:
        console.print(f"{tag('analyze')} No cluster syntheses found, skipping global synthesis.")
        return None

    condensed = _condense_for_global(cluster_syntheses)

    console.print(
        f"{tag('analyze')} Synthesizing global insights across {len(condensed)} clusters..."
    )
    with console.status(f"{tag('analyze')} Global synthesis..."):
        result = synthesize_analyses(
            config,
            condensed,
            system_prompt=GLOBAL_SYSTEM_PROMPT,
            narrative_prompt=GLOBAL_NARRATIVE_PROMPT,
            signals_prompt=GLOBAL_SIGNALS_PROMPT,
            tensions_prompt=GLOBAL_TENSIONS_PROMPT,
            actions_prompt=GLOBAL_ACTIONS_PROMPT,
            condense=False,
        )

    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    console.print(f"{tag('analyze')} Global synthesis saved to {out_path}")
    return out_path
