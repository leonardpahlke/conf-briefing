"""Phase 2: Per-section LLM drafting with parallel execution."""

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from conf_briefing.analyze.llm import query_llm_json
from conf_briefing.config import Config
from conf_briefing.console import console, progress_bar, tag
from conf_briefing.io import load_json_file
from conf_briefing.report.schemas import SectionDraft

# --- Post-processing: reconcile fabricated titles to real ones ---


def _tokenize(text: str) -> set[str]:
    """Extract lowercase word tokens, dropping short words."""
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2}


def _best_match(candidate: str, real_titles: set[str]) -> str | None:
    """Find the real talk title with the highest word overlap to candidate.

    Returns the real title if overlap is above threshold, else None.
    """
    candidate_tokens = _tokenize(candidate)
    if len(candidate_tokens) < 2:
        return None

    best_title = None
    best_score = 0.0

    for real in real_titles:
        real_tokens = _tokenize(real)
        if not real_tokens:
            continue
        overlap = len(candidate_tokens & real_tokens)
        # Jaccard-like score weighted toward the candidate (shorter string)
        score = overlap / max(len(candidate_tokens), 1)
        if score > best_score:
            best_score = score
            best_title = real

    # Require >= 50% of the candidate's words to appear in the real title
    if best_score >= 0.5 and best_title:
        return best_title
    return None


def _reconcile_citations(
    citations: list[dict], real_titles: set[str],
) -> tuple[list[dict], int, int]:
    """Reconcile citation talk_titles to real titles via fuzzy matching.

    Returns (reconciled_citations, matched_count, rejected_count).
    """
    reconciled = []
    matched = 0
    rejected = 0

    for c in citations:
        cited = c.get("talk_title", "")
        if cited in real_titles:
            reconciled.append(c)
            matched += 1
            continue

        real = _best_match(cited, real_titles)
        if real:
            c["talk_title"] = real
            reconciled.append(c)
            matched += 1
        else:
            rejected += 1

    return reconciled, matched, rejected


def _best_match_strict(candidate: str, real_titles: set[str]) -> str | None:
    """Stricter matching for prose: requires more overlap to avoid false positives.

    Used for prose reconciliation where bold/italic text could be topic labels,
    not just talk references.
    """
    candidate_tokens = _tokenize(candidate)
    if len(candidate_tokens) < 4:
        return None

    best_title = None
    best_score = 0.0
    best_overlap = 0

    for real in real_titles:
        real_tokens = _tokenize(real)
        if not real_tokens:
            continue
        overlap = len(candidate_tokens & real_tokens)
        score = overlap / max(len(candidate_tokens), 1)
        if score > best_score:
            best_score = score
            best_title = real
            best_overlap = overlap

    # Require >= 50% token overlap AND at least 3 shared words
    if best_score >= 0.5 and best_overlap >= 3 and best_title:
        return best_title
    return None


def _reconcile_prose_titles(prose: str, real_titles: set[str]) -> tuple[str, int]:
    """Replace paraphrased talk titles in prose with real ones.

    Finds bold (**title**) and italic (*title*) references that look like
    talk titles, fuzzy-matches them, and substitutes the real title.
    Uses strict matching to avoid replacing topic labels with talk titles.
    Returns (fixed_prose, substitution_count).
    """
    subs = 0

    def _replace_match(m: re.Match) -> str:
        nonlocal subs
        delim = m.group(1)  # ** or *
        candidate = m.group(2)

        # Skip short strings — likely topic labels, not talk references
        if len(candidate) < 30:
            return m.group(0)

        # Already a real title?
        if candidate in real_titles:
            return m.group(0)

        real = _best_match_strict(candidate, real_titles)
        if real:
            subs += 1
            return f"{delim}{real}{delim}"
        return m.group(0)

    # Match **bold** references (non-greedy, 30+ chars)
    prose = re.sub(r"(\*\*)((?:[^*]|\*(?!\*)){30,}?)\1", _replace_match, prose)
    # Match *italic* references (non-greedy, 30+ chars, not inside bold)
    prose = re.sub(r"(?<!\*)(\*)((?:[^*]){30,}?)\1(?!\*)", _replace_match, prose)

    # Match *"quoted italic"* references (landscape section uses this style)
    def _replace_quoted(m: re.Match) -> str:
        nonlocal subs
        candidate = m.group(1)
        if len(candidate) < 20 or candidate in real_titles:
            return m.group(0)
        real = _best_match_strict(candidate, real_titles)
        if real:
            subs += 1
            return f'*"{real}"*'
        return m.group(0)

    prose = re.sub(
        r'\*["\u201c]([^""\u201d]{20,}?)["\u201d]\*',
        _replace_quoted,
        prose,
    )

    return prose, subs


# --- Post-processing: scrub fabricated metrics and speakers ---


def _build_source_metrics(data: dict) -> set[str]:
    """Collect all percentages present in pipeline source data."""
    source_pcts: set[str] = set()
    # Per-talk fields
    for talk in data.get("talks", []):
        for field in ("summary", "key_takeaways", "problems_discussed"):
            val = talk.get(field, "")
            if isinstance(val, list):
                val = " ".join(str(v) for v in val)
            source_pcts.update(re.findall(r"\d+(?:\.\d+)?%", str(val)))
    # Cluster synthesis narratives
    for cluster in data.get("clusters", []):
        for field in ("narrative", "common_problems"):
            val = cluster.get(field, "")
            if isinstance(val, list):
                val = " ".join(str(v) for v in val)
            source_pcts.update(re.findall(r"\d+(?:\.\d+)?%", str(val)))
    # Global synthesis
    recordings = data.get("recordings", {})
    for field in ("narrative",):
        val = recordings.get(field, "")
        source_pcts.update(re.findall(r"\d+(?:\.\d+)?%", str(val)))
    return source_pcts


def _scrub_ungrounded_metrics(
    prose: str, source_metrics: set[str],
) -> tuple[str, int]:
    """Replace fabricated percentages and numbers with qualitative language.

    Handles both percentages (N%) and specific numeric claims (N hours, Nms, etc.).
    Returns (scrubbed_prose, replacement_count).
    """
    scrubbed = 0

    # --- Pass 1: Percentages ---
    # Patterns ordered most-specific first to prevent partial matches.
    pct_patterns = [
        # "only N% of" → "only a small fraction of"
        (r"only\s+\d+(?:\.\d+)?%\s+of\s+", lambda m, _: "only a small fraction of "),
        # "only N%" → "only a small share"
        (r"only\s+\d+(?:\.\d+)?%", lambda m, _: "only a small share"),
        # "by N%" → "significantly"
        (r"by\s+\d+(?:\.\d+)?%", lambda m, _: "significantly"),
        # "achieved N%" → "achieved high"
        (r"achieved\s+\d+(?:\.\d+)?%", lambda m, _: "achieved high"),
        # "N% of X" → "many X" (drop the "of")
        (r"\d+(?:\.\d+)?%\s+of\s+", lambda m, _: "many "),
        # "N% reduction/decrease/savings" → "significant reduction/..."
        (
            r"\d+(?:\.\d+)?%\s+(reduction|decrease|savings?)",
            lambda m, _: f"significant {m.group(1)}",
        ),
        # "N% faster/increase/improvement" → "notable faster/..."
        (
            r"\d+(?:\.\d+)?%\s+(faster|increase|improvement|more)",
            lambda m, _: f"notable {m.group(1)}",
        ),
        # "(N% in" → "(a significant proportion in"
        (r"\(\d+(?:\.\d+)?%\s+in\s+", lambda m, _: "(a significant proportion in "),
        # "N% in" → "a significant proportion in"
        (r"\d+(?:\.\d+)?%\s+in\s+", lambda m, _: "a significant proportion in "),
        # Fallback: bare "N%" → "significantly"
        (r"\d+(?:\.\d+)?%", lambda m, _: "significantly"),
    ]

    for pattern, replacement in pct_patterns:
        def _do_replace(m: re.Match, _pat=pattern, _rep=replacement) -> str:
            nonlocal scrubbed
            # Extract just the percentage from the match to check source
            pct_match = re.search(r"\d+(?:\.\d+)?%", m.group())
            if pct_match and pct_match.group() in source_metrics:
                return m.group()  # keep grounded metrics
            scrubbed += 1
            return _rep(m, None)

        prose = re.sub(pattern, _do_replace, prose)

    # --- Pass 2: Non-percentage specific numbers ---
    # Build source numbers set (same sources as percentages)
    source_numbers = set()
    for pct in source_metrics:
        source_numbers.add(pct.rstrip("%"))
    # Also collect raw numbers from source text
    source_text = " ".join(str(v) for v in source_metrics)
    source_numbers.update(re.findall(r"\d+", source_text))

    num_patterns = [
        # "N hours" → "hours" (remove specific number)
        (r"\d+\s+hours?", "hours"),
        # "N minutes" → "minutes"
        (r"\d+\s+minutes?", "minutes"),
        # "Nms" or "N ms" → "sub-second"
        (r"\d+\s*ms\b", "sub-second"),
        # "N-N months" → "several months"
        (r"\d+[\s\u2013-]+\d+\s+months?", "several months"),
        # "N months" → "several months"
        (r"\d+\s+months?", "several months"),
        # "N seconds" → "seconds"
        (r"\d+[\s\u2013-]*\d*\s*seconds?", "seconds"),
    ]

    for pattern, replacement in num_patterns:
        for m in reversed(list(re.finditer(pattern, prose))):
            # Extract the number to check against source
            num = re.search(r"\d+", m.group())
            if num and num.group() in source_numbers:
                continue  # grounded
            prose = prose[:m.start()] + replacement + prose[m.end():]
            scrubbed += 1

    return prose, scrubbed


def _build_speaker_set(data: dict) -> set[str]:
    """Build a set of known speaker names from schedule data."""
    speakers: set[str] = set()
    for session in data.get("_schedule", []):
        for sp in session.get("speakers", []):
            name = sp.get("name", "").strip()
            if name:
                speakers.add(name.lower())
                # Also add individual name parts for partial matching
                parts = name.split()
                if len(parts) >= 2:
                    speakers.add(parts[-1].lower())  # last name
    return speakers


def _scrub_fabricated_speakers(
    prose: str, known_speakers: set[str],
) -> tuple[str, int]:
    """Remove fabricated speaker attributions from prose.

    Returns (scrubbed_prose, removal_count).
    """
    scrubbed = 0
    # Patterns: "speaker X noted", "X cautioned", "Dr. X's framework"
    attr_patterns = [
        # "speaker Dr. Lena Martinez cautioned that..."
        (
            r"(?:,\s*)?(?:though\s+)?speaker\s+"
            r"(?:(?:Dr|Prof)\.?\s+)?"
            r"[A-Z][a-zà-ÿ]+(?:\s+[A-Z][a-zà-ÿ]+)+"
            r"\s+(?:cautioned|noted|emphasized|argued|stated|observed|highlighted)",
            lambda m: ", though speakers cautioned",
        ),
        # "presented by Dr. Lena Martinez, co-founder"
        (
            r"(?:presented by|by)\s+"
            r"(?:(?:Dr|Prof)\.?\s+)?"
            r"[A-Z][a-zà-ÿ]+(?:\s+[A-Z][a-zà-ÿ]+)+"
            r"(?:\s*,\s*(?:co-founder|engineer|architect|lead|director|CTO|CEO|VP)[^,.)]*)?",
            None,  # needs per-match check
        ),
        # "Dr. Lena Müller's framework" — possessive attribution
        (
            r"(?:Dr|Prof)\.?\s+"
            r"[A-Z][a-zà-ÿ]+(?:\s+[A-Z][a-zà-ÿ]+)+"
            r"['\u2019]s\s+\w+",
            None,  # needs per-match check
        ),
    ]

    for pattern, fixed_replacement in attr_patterns:
        for m in re.finditer(pattern, prose):
            # Extract the name from the match
            name_match = re.search(
                r"(?:(?:Dr|Prof)\.?\s+)?([A-Z][a-zà-ÿ]+(?:\s+[A-Z][a-zà-ÿ]+)+)",
                m.group(),
            )
            if not name_match:
                continue
            name = name_match.group(1)
            name_lower = name.lower()
            # Check if any part matches a known speaker
            if any(part in known_speakers for part in name_lower.split()):
                continue
            if name_lower in known_speakers:
                continue

            scrubbed += 1
            if fixed_replacement and callable(fixed_replacement):
                prose = prose.replace(m.group(), fixed_replacement(m), 1)
            else:
                # Remove the attribution phrase
                prose = prose.replace(m.group(), "", 1)
                # Clean up resulting double spaces or orphaned commas
                prose = re.sub(r"  +", " ", prose)
                prose = re.sub(r" ,", ",", prose)

    return prose, scrubbed


SYSTEM_PROMPT = """\
You are a technical analyst writing a conference briefing for a cloud-native \
engineering team. Write clear, direct prose. Use ### subheadings to break the \
section into 2-4 logical parts. Mix prose paragraphs with short lists (3-5 items) \
where they improve scannability. Vary your opening — do NOT start with \
"KubeCon EU 2026 revealed a critical..." or similar generic openers. \
Reference talks by their exact title as given in the data. Attribute claims to \
speakers using only the names provided in the data — do not invent speaker names. \
Distinguish production evidence from theoretical claims or vendor demos. \
Do not invent specific metrics, percentages, or numbers not present in the \
provided analyses. When you want to quantify something but lack a specific \
number from the data, use qualitative language (e.g., "significant reduction", \
"roughly half") instead of inventing a percentage."""

# --- Per-section-type prompts ---

DEEP_DIVE_PROMPT = """\
Write a deep-dive analysis section for the topic cluster "{title}" from {conference_name}.

{guidance}

Here is the cluster synthesis (narrative, themes, tensions, technologies):
{cluster_synthesis}

Here are the individual talk analyses for this cluster:
{talks_json}

IMPORTANT — these are the ONLY valid talk titles for citations. Use exact titles:
{available_titles}

Write ~{word_budget} words of analytical markdown. Requirements:
- Start with a specific, concrete finding — NOT a general statement about the conference.
- Use 2-3 markdown subheadings (###) to organize the analysis logically.
- Reference talks using their exact titles from the data above.
- Attribute claims to speakers only if their names appear in the talk data.
- Highlight where production evidence differs from theoretical claims.
- Call out the 2-3 most important talks worth watching and why.
- End with a concise "what this means for practitioners" paragraph.
- Do NOT use cliches like "three themes emerged" or "the most impactful talks were."

Also return:
- citations: For each specific claim that references a talk, include a citation \
using the exact talk title from the data. Do not paraphrase or shorten talk titles. \
Set needs_quote=true for quantitative claims or strong assertions that would benefit \
from a verbatim transcript quote.
- key_takeaway: A single sentence capturing this section's main insight."""

BRIEF_PROMPT = """\
Write a brief overview for the topic cluster "{title}" from {conference_name}.

Cluster synthesis:
{cluster_synthesis}

Recommended talks: {recommended_talks}

Write ~{word_budget} words. Name the key technology or trend and the top talk \
worth watching (use exact talk titles from the data). If the data is thin, write \
a short honest summary — do NOT pad with generic statements or say "no specific \
talks were highlighted." If there is little to say, say what there is and stop.

Return citations using exact talk titles (needs_quote=false for briefs) \
and a key_takeaway sentence."""

LANDSCAPE_PROMPT = """\
Write a conference landscape overview for {conference_name}.

Conference narrative:
{narrative}

Top keywords: {keywords}

Track distribution:
{track_distribution}

Company presence (top 20):
{company_presence}

Total talks analyzed: {talk_count}
Number of topic clusters: {cluster_count}

Write ~{word_budget} words of markdown. Use ### subheadings. Cover: \
what topics dominated, which companies were most active, how tracks were distributed, \
and what the conference's center of gravity was. Start with the most surprising or \
important finding, not a generic overview.

Return an empty citations list and a key_takeaway sentence."""

CROSS_CUTTING_PROMPT = """\
Write a cross-cutting themes section for {conference_name}.

{prior_chapters_note}

Themes that span multiple clusters:
{themes_json}

Write ~{word_budget} words of markdown with a ### subheading per theme. For each \
theme, explain how it manifests across different topic areas and why it matters. \
Draw connections between seemingly unrelated clusters. Reference earlier chapters \
by name instead of repeating their analysis.

Return citations for claims referencing specific talks (needs_quote=false) and a key_takeaway."""

TENSIONS_PROMPT = """\
Write a tensions and debates section for {conference_name}.

The following chapters provide additional context: {prior_chapter_titles}
You may reference them, but write a COMPLETE analysis of each tension.

Tensions identified across clusters:
{tensions_json}

Write a FULL section of at least {word_budget} words of markdown with a ### subheading \
per tension. For each tension, write 1-2 substantial paragraphs that:
- Frame it as a genuine debate with legitimate arguments on both sides
- Cite specific talks or technologies on each side
- Note the severity and practical implications for practitioners
Do NOT abbreviate — each tension deserves thorough treatment.

Return citations (needs_quote=true for strong opposing claims) and a key_takeaway."""

MATURITY_PROMPT = """\
Write a technology maturity assessment section for {conference_name}.

{prior_chapters_note}

Maturity landscape (technology radar — entries across ALL rings):
{maturity_json}

Write ~{word_budget} words of markdown with a ### subheading per ring \
(Adopt, Trial, Assess, Hold). Prioritize DEPTH over breadth:
- **Adopt**: Pick the 3-4 most impactful technologies. For each, write 2-3 \
sentences explaining the production evidence and what adopters should know.
- **Trial**: Pick 2-3. Explain what validation is still needed.
- **Assess**: Pick 2-3. Explain what blocks adoption.
- **Hold**: Name anything to deprioritize and why.

If you can only name a technology but cannot explain the evidence for its \
placement, omit it. A reader should finish this section knowing exactly what \
to invest in and what to wait on.

Return citations (needs_quote=false) and a key_takeaway."""

ACTIONS_PROMPT = """\
Write a recommended actions section for {conference_name}.

{prior_chapters_note}

Recommended actions grouped by readiness level:
{actions_json}

Evaluation topics the team cares about:
{eval_topics}

Write ~{word_budget} words of markdown with these ### subheadings:
- **Start Now**: 3-4 actions backed by production evidence. Low adoption risk.
- **Evaluate**: 2-3 actions with strong signal but needing validation in your env.
- **Track**: 1-2 emerging directions not ready for production yet.

Each action heading should be the ACTION ITSELF (e.g., "Adopt Kyverno MCP for \
multi-cluster governance"), NOT a talk title. Under each, state what to do, why, \
and what evidence supports it. Consolidate similar actions from different clusters.

Return citations (needs_quote=false) and a key_takeaway."""


def _prior_chapters_note(data: dict) -> str:
    """Build a note listing deep-dive chapters for repetition reduction (2D)."""
    outline_sections = data.get("_outline_sections", [])
    titles = [
        s["title"] for s in outline_sections
        if s.get("section_type") in ("cluster_deep_dive", "cluster_brief")
    ]
    if not titles:
        return ""
    title_list = "\n".join(f"- {t}" for t in titles)
    return (
        f"These topics have dedicated chapters earlier in this report:\n"
        f"{title_list}\n"
        f"Reference them by name rather than repeating their analysis. "
        f"Focus on NEW synthesis and connections."
    )


def _load_analysis_data(config: Config) -> dict:
    """Load all analysis files needed for drafting."""
    data_dir = config.data_dir
    result = {
        "agenda": load_json_file(data_dir / "analysis_agenda.json")
        if (data_dir / "analysis_agenda.json").exists()
        else {},
        "ranking": load_json_file(data_dir / "analysis_ranking.json")
        if (data_dir / "analysis_ranking.json").exists()
        else [],
        "clusters": load_json_file(data_dir / "analysis_clusters.json")
        if (data_dir / "analysis_clusters.json").exists()
        else [],
        "talks": load_json_file(data_dir / "analysis_talks.json")
        if (data_dir / "analysis_talks.json").exists()
        else [],
        "recordings": load_json_file(data_dir / "analysis_recordings.json")
        if (data_dir / "analysis_recordings.json").exists()
        else {},
    }
    # Load schedule for speaker validation
    schedule_path = data_dir / "schedule_clean.json"
    result["_schedule"] = (
        load_json_file(schedule_path) if schedule_path.exists() else []
    )
    return result


def _condense_talk(talk: dict) -> dict:
    """Trim a talk analysis to essential fields for section drafting."""
    keep = [
        "title", "speakers", "summary", "key_takeaways", "tools_and_projects",
        "problems_discussed", "evidence_quality", "speaker_perspective",
        "maturity_assessments", "caveats_and_concerns", "technology_stance",
    ]
    return {k: talk[k] for k in keep if k in talk}


def _find_cluster_synthesis(clusters: list[dict], cluster_name: str) -> dict:
    """Find cluster synthesis by name."""
    for cs in clusters:
        if cs.get("cluster_name") == cluster_name:
            return cs
    return {}


def _find_cluster_talks(
    agenda: dict, talks: list[dict], cluster_name: str,
) -> list[dict]:
    """Find and condense talk analyses belonging to a cluster."""
    # Get talk titles from agenda cluster
    cluster_talk_titles = set()
    for cluster in agenda.get("clusters", []):
        if cluster.get("name") == cluster_name:
            cluster_talk_titles = set(cluster.get("talks", []))
            break

    talks_by_title = {t["title"]: t for t in talks if t.get("title")}
    # Prefix matching for fuzzy title resolution
    talks_by_prefix = {}
    for title, talk in talks_by_title.items():
        prefix = title.split(" - ")[0].strip()
        if prefix not in talks_by_prefix:
            talks_by_prefix[prefix] = talk

    result = []
    for title in cluster_talk_titles:
        talk = talks_by_title.get(title)
        if not talk:
            prefix = title.split(" - ")[0].strip()
            talk = talks_by_prefix.get(prefix)
        if not talk:
            for full_title, t in talks_by_title.items():
                if full_title.startswith(title):
                    talk = t
                    break
        if talk:
            result.append(_condense_talk(talk))
    return result


def _find_recommended_talks(ranking: list[dict], cluster_name: str) -> list[str]:
    """Get recommended talk titles for a cluster from the ranking."""
    for entry in ranking:
        if entry.get("name") == cluster_name:
            return entry.get("recommended_talks", [])
    return []


def _extract_company_name(role_string: str) -> str:
    """Extract company name from role strings like 'Software Engineer, Google'."""
    if not role_string:
        return role_string
    # Common pattern: "Title, Company" or just "Company"
    parts = role_string.rsplit(", ", 1)
    if len(parts) == 2:
        return parts[1].strip()
    return role_string.strip()


def _draft_section(
    config: Config, section: dict, data: dict,
) -> dict | None:
    """Draft a single section via LLM call."""
    section_type = section["section_type"]
    title = section["title"]
    cluster_name = section.get("cluster_name", "")
    word_budget = section.get("word_budget", 500)
    guidance = section.get("guidance", "")
    conf_name = config.conference.name

    if section_type == "cluster_deep_dive":
        cs = _find_cluster_synthesis(data["clusters"], cluster_name)
        cluster_talks = _find_cluster_talks(
            data["agenda"], data["talks"], cluster_name
        )
        available_titles = [t["title"] for t in cluster_talks if t.get("title")]
        prompt = DEEP_DIVE_PROMPT.format(
            title=title,
            conference_name=conf_name,
            guidance=f"Editorial guidance: {guidance}" if guidance else "",
            cluster_synthesis=json.dumps(cs, indent=2, ensure_ascii=False),
            talks_json=json.dumps(cluster_talks, indent=2, ensure_ascii=False),
            available_titles="\n".join(f"- {t}" for t in available_titles),
            word_budget=word_budget,
        )
        max_tokens = 6144

    elif section_type == "cluster_brief":
        cs = _find_cluster_synthesis(data["clusters"], cluster_name)
        rec_talks = _find_recommended_talks(data["ranking"], cluster_name)
        prompt = BRIEF_PROMPT.format(
            title=title,
            conference_name=conf_name,
            cluster_synthesis=json.dumps(
                {k: cs[k] for k in ["narrative", "cross_cutting_themes"] if k in cs},
                indent=2, ensure_ascii=False,
            ),
            recommended_talks=", ".join(rec_talks) if rec_talks else "None specified",
            word_budget=word_budget,
        )
        max_tokens = 2048

    elif section_type == "landscape":
        agenda = data["agenda"]
        # Clean company presence
        raw_presence = agenda.get("company_presence", {})
        clean_presence = {}
        for role_str, count in raw_presence.items():
            company = _extract_company_name(role_str)
            clean_presence[company] = clean_presence.get(company, 0) + count
        # Sort and take top 20
        sorted_companies = sorted(
            clean_presence.items(), key=lambda x: x[1], reverse=True
        )[:20]

        prompt = LANDSCAPE_PROMPT.format(
            conference_name=conf_name,
            narrative=agenda.get("narrative", ""),
            keywords=", ".join(agenda.get("top_keywords", [])[:20]),
            track_distribution=json.dumps(
                agenda.get("track_distribution", {}), indent=2
            ),
            company_presence=json.dumps(
                dict(sorted_companies), indent=2
            ),
            talk_count=len(data["talks"]),
            cluster_count=len(agenda.get("clusters", [])),
            word_budget=word_budget,
        )
        max_tokens = 4096

    elif section_type == "cross_cutting":
        recordings = data["recordings"]
        prompt = CROSS_CUTTING_PROMPT.format(
            conference_name=conf_name,
            prior_chapters_note=_prior_chapters_note(data),
            themes_json=json.dumps(
                recordings.get("cross_cutting_themes", []),
                indent=2, ensure_ascii=False,
            ),
            word_budget=word_budget,
        )
        max_tokens = 4096

    elif section_type == "tensions":
        recordings = data["recordings"]
        chapter_titles = [
            s["title"] for s in data.get("_outline_sections", [])
            if s.get("section_type") in ("cluster_deep_dive", "cluster_brief")
        ]
        prompt = TENSIONS_PROMPT.format(
            conference_name=conf_name,
            prior_chapter_titles=", ".join(chapter_titles) if chapter_titles else "none",
            tensions_json=json.dumps(
                recordings.get("tensions", []),
                indent=2, ensure_ascii=False,
            ),
            word_budget=word_budget,
        )
        max_tokens = 4096

    elif section_type == "maturity":
        # Aggregate maturity data from all clusters (richer than global summary)
        evidence_rank = {
            "production_proven": 4, "benchmarked": 3, "case_study": 2, "anecdotal": 1,
        }
        by_tech: dict[str, dict] = {}
        for cluster in data.get("clusters", []):
            for item in cluster.get("maturity_landscape", []):
                tech = item.get("technology", "")
                if not tech:
                    continue
                rank = evidence_rank.get(item.get("evidence_quality", ""), 0)
                if tech not in by_tech or rank > evidence_rank.get(
                    by_tech[tech].get("evidence_quality", ""), 0,
                ):
                    by_tech[tech] = {**item, "source_cluster": cluster.get("cluster_name", "")}
        maturity_data = sorted(by_tech.values(), key=lambda x: x.get("ring", "assess"))
        # Fall back to global data if clusters have nothing
        if not maturity_data:
            maturity_data = data["recordings"].get("maturity_landscape", [])
        prompt = MATURITY_PROMPT.format(
            conference_name=conf_name,
            prior_chapters_note=_prior_chapters_note(data),
            maturity_json=json.dumps(maturity_data, indent=2, ensure_ascii=False),
            word_budget=word_budget,
        )
        max_tokens = 4096

    elif section_type == "actions":
        # Aggregate actions from all clusters for better urgency coverage
        by_urgency: dict[str, list] = {
            "immediate": [], "next_quarter": [], "long_term": [],
        }
        for cluster in data.get("clusters", []):
            for action in cluster.get("recommended_actions", []):
                urgency = action.get("urgency", "immediate")
                entry = {**action, "source_cluster": cluster.get("cluster_name", "")}
                by_urgency.setdefault(urgency, []).append(entry)
        # Fall back to global data if clusters have nothing
        if not any(by_urgency.values()):
            for action in data["recordings"].get("recommended_actions", []):
                urgency = action.get("urgency", "immediate")
                by_urgency.setdefault(urgency, []).append(action)
        prompt = ACTIONS_PROMPT.format(
            conference_name=conf_name,
            prior_chapters_note=_prior_chapters_note(data),
            actions_json=json.dumps(by_urgency, indent=2, ensure_ascii=False),
            eval_topics="\n".join(f"- {t}" for t in config.analyze.eval_topics),
            word_budget=word_budget,
        )
        max_tokens = 4096
    else:
        console.print(
            f"  {tag('report')} Unknown section type '{section_type}', skipping."
        )
        return None

    result = query_llm_json(
        config, SYSTEM_PROMPT, prompt, max_tokens=max_tokens, schema=SectionDraft
    )

    # Overwrite section_id and title from outline (avoid LLM paraphrasing)
    result["section_id"] = section["section_id"]
    result["title"] = title

    # Post-process: reconcile paraphrased titles to real ones
    all_talk_titles = {t["title"] for t in data["talks"] if t.get("title")}

    # Reconcile citations
    citations = result.get("citations", [])
    if citations:
        reconciled, matched, rejected = _reconcile_citations(citations, all_talk_titles)
        result["citations"] = reconciled
        if matched or rejected:
            console.print(
                f"  {tag('report')} {title[:40]} — citations: "
                f"{matched} reconciled, {rejected} rejected"
            )

    # Reconcile talk titles in prose
    prose, subs = _reconcile_prose_titles(result.get("prose", ""), all_talk_titles)
    if subs:
        result["prose"] = prose
        console.print(
            f"  {tag('report')} {title[:40]} — {subs} title(s) reconciled in prose"
        )

    # Scrub ungrounded metrics
    source_metrics = _build_source_metrics(data)
    result["prose"], metric_scrubs = _scrub_ungrounded_metrics(
        result["prose"], source_metrics,
    )
    if metric_scrubs:
        console.print(
            f"  {tag('report')} {title[:40]} — "
            f"scrubbed {metric_scrubs} ungrounded metric(s)"
        )

    # Scrub fabricated speakers
    known_speakers = _build_speaker_set(data)
    result["prose"], speaker_scrubs = _scrub_fabricated_speakers(
        result["prose"], known_speakers,
    )
    if speaker_scrubs:
        console.print(
            f"  {tag('report')} {title[:40]} — "
            f"scrubbed {speaker_scrubs} fabricated speaker(s)"
        )

    return result


def draft_sections(config: Config, outline: dict) -> list[dict]:
    """Draft all report sections in parallel (Phase 2).

    Returns list of section draft dicts. Saves checkpoint incrementally.
    """
    data_dir = config.data_dir
    reports_dir = data_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "report_sections.json"
    outline_path = reports_dir / "report_outline.json"

    sections = outline.get("sections", [])
    if not sections:
        console.print(f"{tag('report')} No sections in outline, skipping drafting.")
        return []

    # Cache check
    if out_path.exists() and outline_path.exists():
        if out_path.stat().st_mtime > outline_path.stat().st_mtime:
            console.print(f"{tag('report')} Section drafts are up-to-date, skipping.")
            return load_json_file(out_path)

    data = _load_analysis_data(config)
    data["_outline_sections"] = sections  # for _prior_chapters_note

    console.print(
        f"{tag('report')} Drafting {len(sections)} sections..."
    )

    drafts: list[dict] = []
    failed = 0

    with progress_bar() as pb:
        task = pb.add_task(
            f"{tag('report')} Drafting sections", total=len(sections)
        )

        with ThreadPoolExecutor(max_workers=config.llm.num_parallel) as executor:
            futures = {
                executor.submit(_draft_section, config, s, data): s
                for s in sections
            }
            for future in as_completed(futures):
                section = futures[future]
                title = section["title"][:50]
                try:
                    result = future.result()
                    if result:
                        drafts.append(result)
                        # Save incrementally
                        out_path.write_text(
                            json.dumps(drafts, indent=2, ensure_ascii=False)
                        )
                    pb.update(task, advance=1, description=f"{tag('report')} {title}")
                except Exception as e:
                    failed += 1
                    pb.update(
                        task, advance=1,
                        description=f"{tag('report')} {title} [red]failed[/red]",
                    )
                    console.print(f"  {tag('report')} [red]{title} — {e}[/red]")

    if failed:
        console.print(
            f"{tag('report')} [yellow]{failed}/{len(sections)} sections failed[/yellow]"
        )

    # Sort by outline priority
    section_order = {s["section_id"]: s.get("priority", 99) for s in sections}
    drafts.sort(key=lambda d: section_order.get(d["section_id"], 99))

    out_path.write_text(json.dumps(drafts, indent=2, ensure_ascii=False))
    console.print(
        f"{tag('report')} {len(drafts)} section drafts saved to {out_path}"
    )
    return drafts
