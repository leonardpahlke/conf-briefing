"""Pydantic models for LLM-driven report generation."""

from typing import Literal

from pydantic import BaseModel, Field


# --- Phase 1: Outline ---


class SectionSpec(BaseModel):
    """Specification for a single report section."""

    section_id: str
    title: str
    section_type: Literal[
        "cluster_deep_dive",
        "cluster_brief",
        "cross_cutting",
        "tensions",
        "maturity",
        "actions",
        "landscape",
    ]
    cluster_name: str = ""
    word_budget: int = 500
    priority: int = 1
    guidance: str = ""
    source_talks: list[str] = Field(default_factory=list)


class ReportOutline(BaseModel):
    """Full report outline produced by Phase 1."""

    conference_name: str
    thesis: str
    sections: list[SectionSpec]
    appendix_strategy: Literal["top_talks_only", "by_cluster", "none"] = (
        "top_talks_only"
    )
    total_word_budget: int = 8000


# --- Phase 2: Section Drafts ---


class Citation(BaseModel):
    """A reference to a specific talk used as evidence."""

    talk_title: str
    claim: str
    needs_quote: bool = False


class SectionDraft(BaseModel):
    """A drafted section of the report."""

    section_id: str
    title: str
    prose: str
    citations: list[Citation] = Field(default_factory=list)
    key_takeaway: str = ""


# --- Phase 3: Evidence Enrichment ---


class EnrichedQuote(BaseModel):
    """A direct quote from a transcript, matched to a citation."""

    talk_title: str
    speaker: str = ""
    quote: str
    timestamp_sec: float = 0.0
    video_id: str = ""


class EnrichedSection(BaseModel):
    """A section with evidence integrated."""

    section_id: str
    title: str
    prose: str
    quotes: list[EnrichedQuote] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    key_takeaway: str = ""


# --- Phase 4: Assembly ---


class ExecutiveSummary(BaseModel):
    """LLM-generated executive summary."""

    summary: str
    key_findings: list[str]
    top_actions: list[str]
