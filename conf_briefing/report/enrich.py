"""Phase 3: Evidence enrichment via RAG-based transcript quote retrieval."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from conf_briefing.analyze.llm import query_llm_json, query_llm
from conf_briefing.config import Config
from conf_briefing.console import console, progress_bar, tag
from conf_briefing.io import load_json_file
from conf_briefing.report.schemas import EnrichedQuote

QUOTE_SYSTEM_PROMPT = """\
You are extracting a verbatim quote from a conference talk transcript. \
Return the most relevant direct quote that supports the given claim."""

QUOTE_EXTRACT_PROMPT = """\
Find a direct quote from this transcript excerpt that supports the claim below.

Claim: {claim}
Talk: {talk_title}

Transcript excerpt:
{transcript_excerpt}

Extract as JSON:
- talk_title: exact talk title as given above
- speaker: the speaker label if identifiable (e.g., "Speaker 1"), or empty string
- quote: the verbatim quote (2-4 sentences max, clean up obvious speech-recognition \
errors but keep the substance)
- timestamp_sec: approximate timestamp in seconds if available, otherwise 0
- video_id: empty string (not available from this context)

If no relevant quote exists in the excerpt, return an empty quote string."""

INTEGRATE_SYSTEM_PROMPT = """\
You are a technical writer integrating evidence quotes into an analytical report section. \
Weave quotes naturally into the existing prose using markdown blockquotes."""

INTEGRATE_PROMPT = """\
Integrate these direct quotes into the report section below. Requirements:
- Use markdown blockquote format (> quote) with speaker attribution
- Place quotes at the most relevant point in the prose
- Do not change the analytical content — only add evidence
- If a quote doesn't fit naturally, skip it
- Keep the section roughly the same length (quotes add, don't replace)

Section prose:
{prose}

Quotes to integrate:
{quotes_json}

Return the updated prose as a single markdown string (no JSON wrapping)."""


_QUERY_INDEX_IMPORT_WARNED = False


def _query_for_quote(
    config: Config, claim: str, talk_title: str,
) -> dict | None:
    """Retrieve a transcript quote for a citation via ChromaDB."""
    global _QUERY_INDEX_IMPORT_WARNED
    try:
        from conf_briefing.query.index import query_index
    except Exception as e:
        if not _QUERY_INDEX_IMPORT_WARNED:
            _QUERY_INDEX_IMPORT_WARNED = True
            console.print(
                f"  {tag('report')} [red]Cannot import query_index: {e}. "
                f"Enrichment will be skipped for all sections. "
                f"Run 'just query <event> \"test\"' first to build the index.[/red]"
            )
        return None

    # Search for transcript segments matching the claim
    hits = query_index(
        config,
        query=claim,
        top_k=8,
        chunk_types=["transcript_segment"],
    )

    # Filter to hits from the cited talk
    matching = [
        h for h in hits
        if talk_title.lower() in h.get("metadata", {}).get("talk_title", "").lower()
    ]

    # Fallback: try slide content if no transcript match
    if not matching:
        hits = query_index(
            config,
            query=claim,
            top_k=5,
            chunk_types=["slide_content"],
        )
        matching = [
            h for h in hits
            if talk_title.lower()
            in h.get("metadata", {}).get("talk_title", "").lower()
        ]

    if not matching:
        return None

    # Use the best match
    best = matching[0]
    excerpt = best["text"]

    # Extract quote via small LLM call
    prompt = QUOTE_EXTRACT_PROMPT.format(
        claim=claim,
        talk_title=talk_title,
        transcript_excerpt=excerpt[:3000],
    )

    result = query_llm_json(
        config, QUOTE_SYSTEM_PROMPT, prompt,
        max_tokens=512, schema=EnrichedQuote,
    )

    # Skip empty quotes
    if not result.get("quote", "").strip():
        return None

    # Add video_id from metadata if available
    video_id = best.get("metadata", {}).get("video_id", "")
    if video_id:
        result["video_id"] = video_id

    # Add timestamp from metadata if available and quote didn't provide one
    if not result.get("timestamp_sec") and best.get("metadata", {}).get("start_time"):
        try:
            result["timestamp_sec"] = float(best["metadata"]["start_time"])
        except (ValueError, TypeError):
            pass

    return result


def _enrich_section(
    config: Config, section: dict, max_quotes: int,
) -> dict:
    """Enrich a single section with transcript quotes."""
    citations = section.get("citations", [])
    needs_quote = [c for c in citations if c.get("needs_quote")]

    if not needs_quote:
        return {
            "section_id": section["section_id"],
            "title": section["title"],
            "prose": section["prose"],
            "quotes": [],
            "citations": citations,
            "key_takeaway": section.get("key_takeaway", ""),
        }

    # Cap the number of quotes
    needs_quote = needs_quote[:max_quotes]

    quotes = []
    for citation in needs_quote:
        result = _query_for_quote(
            config, citation["claim"], citation["talk_title"],
        )
        if result:
            quotes.append(result)

    if not quotes:
        return {
            "section_id": section["section_id"],
            "title": section["title"],
            "prose": section["prose"],
            "quotes": [],
            "citations": citations,
            "key_takeaway": section.get("key_takeaway", ""),
        }

    # Integration pass: weave quotes into prose via LLM
    quotes_for_prompt = [
        {
            "speaker": q.get("speaker", ""),
            "quote": q["quote"],
            "talk_title": q["talk_title"],
        }
        for q in quotes
    ]

    prompt = INTEGRATE_PROMPT.format(
        prose=section["prose"],
        quotes_json=json.dumps(quotes_for_prompt, indent=2, ensure_ascii=False),
    )

    enriched_prose = query_llm(
        config, INTEGRATE_SYSTEM_PROMPT, prompt, max_tokens=6144,
    )

    return {
        "section_id": section["section_id"],
        "title": section["title"],
        "prose": enriched_prose,
        "quotes": quotes,
        "citations": citations,
        "key_takeaway": section.get("key_takeaway", ""),
    }


def enrich_sections(
    config: Config, sections: list[dict], outline: dict,
) -> list[dict]:
    """Enrich sections with transcript evidence (Phase 3).

    Only enriches cluster_deep_dive sections. Others pass through as-is.
    Returns list of enriched section dicts.
    """
    data_dir = config.data_dir
    reports_dir = data_dir / "reports"
    out_path = reports_dir / "report_sections_enriched.json"
    sections_path = reports_dir / "report_sections.json"

    def _pass_through(s: dict) -> dict:
        """Convert a draft section to enriched format, preserving all fields."""
        return {
            "section_id": s["section_id"],
            "title": s["title"],
            "prose": s["prose"],
            "quotes": [],
            "citations": s.get("citations", []),
            "key_takeaway": s.get("key_takeaway", ""),
        }

    if config.report.skip_enrichment:
        console.print(f"{tag('report')} Enrichment skipped (skip_enrichment=true).")
        result = [_pass_through(s) for s in sections]
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        return result

    # Cache check
    if out_path.exists() and sections_path.exists():
        if out_path.stat().st_mtime > sections_path.stat().st_mtime:
            console.print(
                f"{tag('report')} Enriched sections are up-to-date, skipping."
            )
            return load_json_file(out_path)

    # Check if ChromaDB index exists
    chroma_dir = data_dir / "chroma"
    if not chroma_dir.exists():
        console.print(
            f"{tag('report')} [yellow]No ChromaDB index found. "
            f"Run 'index' first for transcript evidence, or set "
            f"skip_enrichment=true. Skipping enrichment.[/yellow]"
        )
        result = [_pass_through(s) for s in sections]
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        return result

    # Identify sections to enrich (only deep dives)
    outline_sections = {s["section_id"]: s for s in outline.get("sections", [])}
    deep_dive_ids = {
        sid
        for sid, spec in outline_sections.items()
        if spec.get("section_type") == "cluster_deep_dive"
    }

    to_enrich = [s for s in sections if s["section_id"] in deep_dive_ids]
    pass_through = [s for s in sections if s["section_id"] not in deep_dive_ids]

    console.print(
        f"{tag('report')} Enriching {len(to_enrich)} deep-dive sections with "
        f"transcript evidence..."
    )

    max_quotes = config.report.max_quotes_per_section
    enriched: list[dict] = []

    with progress_bar() as pb:
        task = pb.add_task(
            f"{tag('report')} Enriching sections", total=len(to_enrich),
        )

        with ThreadPoolExecutor(max_workers=config.llm.num_parallel) as executor:
            futures = {
                executor.submit(_enrich_section, config, s, max_quotes): s
                for s in to_enrich
            }
            for future in as_completed(futures):
                section = futures[future]
                title = section["title"][:50]
                try:
                    result = future.result()
                    enriched.append(result)
                    quote_count = len(result.get("quotes", []))
                    pb.update(
                        task, advance=1,
                        description=f"{tag('report')} {title} ({quote_count} quotes)",
                    )
                except Exception as e:
                    # Fallback: use draft as-is
                    enriched.append(_pass_through(section))
                    pb.update(
                        task, advance=1,
                        description=f"{tag('report')} {title} [yellow]fallback[/yellow]",
                    )
                    console.print(
                        f"  {tag('report')} [yellow]{title} — enrichment failed: "
                        f"{e}, using draft[/yellow]"
                    )

    # Combine enriched deep dives with pass-through sections
    pass_through_enriched = [_pass_through(s) for s in pass_through]

    all_sections = enriched + pass_through_enriched

    # Restore outline order
    section_order = {
        s["section_id"]: s.get("priority", 99)
        for s in outline.get("sections", [])
    }
    all_sections.sort(key=lambda s: section_order.get(s["section_id"], 99))

    out_path.write_text(json.dumps(all_sections, indent=2, ensure_ascii=False))
    console.print(
        f"{tag('report')} {len(enriched)} enriched + {len(pass_through)} "
        f"pass-through sections saved to {out_path}"
    )
    return all_sections
