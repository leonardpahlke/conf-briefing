"""Post-assembly validation: detect fabricated content in report prose."""

import re

from conf_briefing.config import Config
from conf_briefing.console import console, tag
from conf_briefing.io import load_json_file


def _normalize(text: str) -> str:
    """Lowercase and strip extra whitespace for fuzzy comparison."""
    return re.sub(r"\s+", " ", text.lower().strip())


def _extract_names_from_prose(prose: str) -> list[str]:
    """Extract likely person names from prose (capitalized multi-word sequences).

    Heuristic: two or more consecutive capitalized words not at sentence start,
    optionally preceded by "Dr." or "Prof.".
    """
    # Match patterns like "Dr. Lena Martínez", "Rajiv Iyer", "Anika Chen"
    pattern = r"(?:(?:Dr|Prof|Mr|Ms)\.?\s+)?(?:[A-Z][a-zà-ÿ]+(?:\s+[A-Z][a-zà-ÿ]+)+)"
    return re.findall(pattern, prose)


def _extract_percentages(prose: str) -> list[str]:
    """Extract percentage claims from prose (e.g. '99.98%', '40%')."""
    return re.findall(r"\d+(?:\.\d+)?%", prose)


def _extract_template_artifacts(prose: str) -> list[str]:
    """Find leftover template placeholders in prose."""
    artifacts = []
    # [needs_quote=true], [needs_quote=false], {placeholder}
    artifacts.extend(re.findall(r"\[needs_quote\s*=\s*\w+\]", prose))
    artifacts.extend(re.findall(r"\{[a-z_]+\}", prose))
    return artifacts


def validate_report(config: Config, sections: list[dict]) -> dict:
    """Validate assembled report sections against source data.

    Returns a dict with:
      - warnings: list of (section_id, message) tuples
      - stats: summary counts
    """
    data_dir = config.data_dir

    # Load source data
    schedule_path = data_dir / "schedule_clean.json"
    talks_path = data_dir / "analysis_talks.json"
    agenda_path = data_dir / "analysis_agenda.json"

    schedule = load_json_file(schedule_path) if schedule_path.exists() else []
    talks = load_json_file(talks_path) if talks_path.exists() else []
    _ = agenda_path  # reserved for future cluster-name validation

    # Build lookup sets
    schedule_speakers = set()
    schedule_companies = set()
    for session in schedule:
        for speaker in session.get("speakers", []):
            name = speaker.get("name", "").strip()
            if name:
                schedule_speakers.add(name)
                # Also add last-name-only for partial matching
                parts = name.split()
                if len(parts) >= 2:
                    schedule_speakers.add(parts[-1])
            company = speaker.get("company", "").strip()
            if company:
                # Extract company name from "Title, Company" format
                company_name = company.rsplit(", ", 1)[-1].strip()
                schedule_companies.add(company_name)

    schedule_companies_lower = {_normalize(c) for c in schedule_companies}

    schedule_speakers_lower = {_normalize(s) for s in schedule_speakers}

    talk_titles = {t["title"] for t in talks if t.get("title")}
    talk_titles_lower = {_normalize(t) for t in talk_titles}

    # Collect all metrics from upstream summaries and takeaways
    source_percentages: set[str] = set()
    for talk in talks:
        for field in ("summary", "key_takeaways", "problems_discussed"):
            value = talk.get(field, "")
            if isinstance(value, list):
                value = " ".join(str(v) for v in value)
            source_percentages.update(re.findall(r"\d+(?:\.\d+)?%", str(value)))

    warnings: list[tuple[str, str]] = []
    stats = {
        "sections_checked": 0,
        "unknown_speakers": 0,
        "ungrounded_metrics": 0,
        "template_artifacts": 0,
        "fabricated_citations": 0,
    }

    for section in sections:
        sid = section.get("section_id", "?")
        prose = section.get("prose", "")
        stats["sections_checked"] += 1

        # 1. Check speaker names in prose against schedule
        # Only check names that appear in attribution context (said, presented,
        # according to, by, from) to reduce false positives from technology
        # names and organization names
        attr_patterns = [
            r"(?:presented by|according to|by|said|argues?|notes?|demonstrates?|shows?)\s+"
            r"(?:(?:Dr|Prof)\.?\s+)?([A-Z][a-zà-ÿ]+(?:\s+[A-Z][a-zà-ÿ]+)+)",
            r"(?:(?:Dr|Prof)\.?\s+)?([A-Z][a-zà-ÿ]+(?:\s+[A-Z][a-zà-ÿ]+)+)"
            r"(?:\s*,\s*(?:co-founder|engineer|architect|lead|director|CTO|CEO|VP))",
            r"\((?:presented by|by)\s+"
            r"(?:(?:Dr|Prof)\.?\s+)?([A-Z][a-zà-ÿ]+(?:\s+[A-Z][a-zà-ÿ]+)+)\)",
        ]
        names_in_prose = set()
        for pat in attr_patterns:
            names_in_prose.update(re.findall(pat, prose))

        for name in names_in_prose:
            normalized = _normalize(name)
            if len(normalized) < 4:
                continue
            # Skip if it's a known company name, not a person
            if any(
                c in normalized or normalized in c
                for c in schedule_companies_lower
            ):
                continue
            if not any(
                known in normalized or normalized in known
                for known in schedule_speakers_lower
            ):
                warnings.append((
                    sid,
                    f"Unknown speaker: \"{name}\" — not in schedule_clean.json",
                ))
                stats["unknown_speakers"] += 1

        # 2. Check percentages against source data
        prose_pcts = _extract_percentages(prose)
        for pct in prose_pcts:
            if pct not in source_percentages:
                warnings.append((
                    sid,
                    f"Ungrounded metric: {pct} — not found in analysis_talks.json",
                ))
                stats["ungrounded_metrics"] += 1

        # 3. Check for template artifacts
        artifacts = _extract_template_artifacts(prose)
        for artifact in artifacts:
            warnings.append((sid, f"Template artifact: {artifact}"))
            stats["template_artifacts"] += 1

        # 4. Check citations reference real talks
        for citation in section.get("citations", []):
            cited = citation.get("talk_title", "")
            if not cited:
                continue
            cited_lower = _normalize(cited)
            # Exact match or prefix match
            if cited_lower not in talk_titles_lower:
                prefix = cited.split(" - ")[0].strip()
                prefix_lower = _normalize(prefix)
                if not any(t.startswith(prefix_lower) for t in talk_titles_lower):
                    warnings.append((
                        sid,
                        f"Fabricated citation: \"{cited[:60]}...\"",
                    ))
                    stats["fabricated_citations"] += 1

    return {"warnings": warnings, "stats": stats}


def run_validation(config: Config, sections: list[dict]) -> None:
    """Run validation and print results to console."""
    result = validate_report(config, sections)
    warnings = result["warnings"]
    stats = result["stats"]

    total_issues = sum(v for k, v in stats.items() if k != "sections_checked")

    if total_issues == 0:
        console.print(
            f"{tag('report')} Validation passed: "
            f"{stats['sections_checked']} sections, no issues."
        )
        return

    console.print(
        f"{tag('report')} [yellow]Validation found {total_issues} issue(s) "
        f"across {stats['sections_checked']} sections:[/yellow]"
    )

    if stats["unknown_speakers"]:
        console.print(
            f"  [yellow]  {stats['unknown_speakers']} unknown speaker name(s)[/yellow]"
        )
    if stats["ungrounded_metrics"]:
        console.print(
            f"  [yellow]  {stats['ungrounded_metrics']} ungrounded metric(s)[/yellow]"
        )
    if stats["template_artifacts"]:
        console.print(
            f"  [red]  {stats['template_artifacts']} template artifact(s)[/red]"
        )
    if stats["fabricated_citations"]:
        console.print(
            f"  [red]  {stats['fabricated_citations']} fabricated citation(s)[/red]"
        )

    # Print first N warnings as examples
    for i, (sid, msg) in enumerate(warnings):
        if i >= 15:
            console.print(f"  ... and {len(warnings) - i} more")
            break
        console.print(f"  [{sid}] {msg}")
