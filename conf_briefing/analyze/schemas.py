"""Pydantic models for structured LLM output."""

from typing import Literal

from pydantic import BaseModel


class MaturityAssessment(BaseModel):
    technology: str = ""
    maturity: Literal["assess", "trial", "adopt", "hold"] = "assess"
    evidence: str = ""


class TechnologyStance(BaseModel):
    technology: str = ""
    stance: Literal["enthusiastic", "cautious", "critical", "neutral"] = "neutral"
    evidence: str = ""


class Relationship(BaseModel):
    entity_a: str = ""
    relation: Literal[
        "replaces", "competes_with", "builds_on", "integrates_with", "extends"
    ] = "builds_on"
    entity_b: str = ""


class TalkAnalysis(BaseModel):
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


# --- Synthesis models ---


class CrossCuttingTheme(BaseModel):
    theme: str = ""
    description: str = ""
    supporting_talks: list[str] = []


class EmergingTechnology(BaseModel):
    technology: str = ""
    mentions: int = 0
    context: str = ""


class TensionSide(BaseModel):
    position: str = ""
    supporting_talks: list[str] = []


class Tension(BaseModel):
    topic: str = ""
    side_a: TensionSide = TensionSide()
    side_b: TensionSide = TensionSide()
    severity: Literal["fundamental", "significant", "minor"] = "minor"
    implication: str = ""


class MaturityLandscapeEntry(BaseModel):
    technology: str = ""
    ring: Literal["assess", "trial", "adopt", "hold"] = "assess"
    evidence_quality: Literal[
        "anecdotal", "case_study", "benchmarked", "production_proven"
    ] = "anecdotal"
    supporting_talks: list[str] = []
    rationale: str = ""


class RecommendedAction(BaseModel):
    action: str = ""
    category: Literal["evaluate", "watch", "talk_to", "adopt", "avoid"] = "evaluate"
    urgency: Literal["immediate", "next_quarter", "long_term"] = "next_quarter"
    supporting_evidence: str = ""


class TechnologyRelationship(BaseModel):
    entity_a: str = ""
    relation: str = ""
    entity_b: str = ""
    supporting_talks: list[str] = []


class SynthesisResult(BaseModel):
    cross_cutting_themes: list[CrossCuttingTheme] = []
    emerging_technologies: list[EmergingTechnology] = []
    common_problems: list[str] = []
    narrative: str = ""
    tensions: list[Tension] = []
    maturity_landscape: list[MaturityLandscapeEntry] = []
    recommended_actions: list[RecommendedAction] = []
    technology_relationships: list[TechnologyRelationship] = []
