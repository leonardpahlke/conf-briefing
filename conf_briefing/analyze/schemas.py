"""Pydantic models for structured LLM output."""

from typing import Literal

from pydantic import BaseModel, Field


# --- Shared sub-models ---


class MaturityAssessment(BaseModel):
    technology: str
    maturity: Literal["assess", "trial", "adopt", "hold"]
    evidence: str


class TechnologyStance(BaseModel):
    technology: str
    stance: Literal["enthusiastic", "cautious", "critical", "neutral"]
    evidence: str


class Relationship(BaseModel):
    entity_a: str
    relation: Literal[
        "replaces", "competes_with", "builds_on", "integrates_with", "extends"
    ]
    entity_b: str


# --- Split talk schemas (no defaults on lists → forces Ollama to produce them) ---


class TalkCore(BaseModel):
    """Call 1: core content extraction."""

    title: str
    summary: str
    key_takeaways: list[str] = Field(min_length=1)
    problems_discussed: list[str] = Field(min_length=1)
    tools_and_projects: list[str] = Field(min_length=1)
    qa_highlights: list[str]
    evidence_quality: Literal[
        "production", "proof_of_concept", "theoretical", "vendor_demo"
    ]
    speaker_perspective: Literal[
        "practitioner", "vendor", "maintainer", "academic"
    ]
    references: list[str]


class TalkSignals(BaseModel):
    """Call 2: signals and relationships."""

    maturity_assessments: list[MaturityAssessment]
    caveats_and_concerns: list[str]
    technology_stance: list[TechnologyStance]
    relationships: list[Relationship]


class TalkAnalysis(BaseModel):
    """Full talk analysis (merged from TalkCore + TalkSignals)."""

    title: str = ""
    key_takeaways: list[str] = []
    qa_highlights: list[str] = []
    problems_discussed: list[str] = []
    tools_and_projects: list[str] = []
    summary: str = ""
    evidence_quality: Literal[
        "production", "proof_of_concept", "theoretical", "vendor_demo"
    ] = "theoretical"
    speaker_perspective: Literal[
        "practitioner", "vendor", "maintainer", "academic"
    ] = "practitioner"
    maturity_assessments: list[MaturityAssessment] = []
    caveats_and_concerns: list[str] = []
    technology_stance: list[TechnologyStance] = []
    relationships: list[Relationship] = []
    references: list[str] = []


# --- Synthesis sub-models (defined before split schemas that reference them) ---


class CrossCuttingTheme(BaseModel):
    theme: str
    description: str
    supporting_talks: list[str]


class EmergingTechnology(BaseModel):
    technology: str
    mentions: int
    context: str


class TechnologyRelationship(BaseModel):
    entity_a: str
    relation: str
    entity_b: str
    supporting_talks: list[str]


class TensionSide(BaseModel):
    position: str
    supporting_talks: list[str]


class Tension(BaseModel):
    topic: str
    side_a: TensionSide
    side_b: TensionSide
    severity: Literal["fundamental", "significant", "minor"]
    implication: str


class MaturityLandscapeEntry(BaseModel):
    technology: str
    ring: Literal["assess", "trial", "adopt", "hold"]
    evidence_quality: Literal[
        "anecdotal", "case_study", "benchmarked", "production_proven"
    ]
    supporting_talks: list[str]
    rationale: str


class RecommendedAction(BaseModel):
    action: str
    category: Literal["evaluate", "watch", "talk_to", "adopt", "avoid"]
    urgency: Literal["immediate", "next_quarter", "long_term"]
    supporting_evidence: str


# --- Split synthesis schemas (no defaults on lists → forces Ollama to produce them) ---


class SynthNarrative(BaseModel):
    """Synthesis call 1: narrative + themes + problems."""

    narrative: str
    cross_cutting_themes: list[CrossCuttingTheme]
    common_problems: list[str] = Field(min_length=1)


class SynthSignals(BaseModel):
    """Synthesis call 2: emerging tech + relationships."""

    emerging_technologies: list[EmergingTechnology]
    technology_relationships: list[TechnologyRelationship]


class SynthTensions(BaseModel):
    """Synthesis call 3: tensions + maturity landscape."""

    tensions: list[Tension]
    maturity_landscape: list[MaturityLandscapeEntry]


class SynthActions(BaseModel):
    """Synthesis call 4: recommended actions."""

    recommended_actions: list[RecommendedAction] = Field(min_length=1)


class SynthesisResult(BaseModel):
    cross_cutting_themes: list[CrossCuttingTheme] = []
    emerging_technologies: list[EmergingTechnology] = []
    common_problems: list[str] = []
    narrative: str = ""
    tensions: list[Tension] = []
    maturity_landscape: list[MaturityLandscapeEntry] = []
    recommended_actions: list[RecommendedAction] = []
    technology_relationships: list[TechnologyRelationship] = []
