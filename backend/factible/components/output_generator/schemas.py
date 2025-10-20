from typing import Dict, List, Literal

from pydantic import BaseModel, Field

from factible.components.claim_extractor.schemas import Claim, ExtractedClaims
from factible.components.online_search.schemas.evidence import EvidenceStance
from factible.components.online_search.schemas.reliability import SiteReliability
from factible.components.transcriptor.schemas import TranscriptData


VerdictConfidence = Literal["low", "medium", "high"]


class EvidenceSourceSummary(BaseModel):
    """Summary of a single evidence source grouped by stance."""

    title: str
    url: str
    reliability: SiteReliability
    stance: EvidenceStance
    evidence_summary: str | None = Field(
        default=None,
        description="Short synthesis of how the source relates to the claim.",
    )
    snippet: str | None = Field(
        default=None,
        description="Optional snippet extracted from the source.",
    )


class ClaimVerdict(BaseModel):
    """LLM-produced conclusion about how evidence relates to a claim."""

    overall_stance: EvidenceStance = Field(
        description="Best overall assessment across 'supports', 'refutes', 'mixed', 'unclear'."
    )
    confidence: VerdictConfidence = Field(
        description="Confidence in the overall stance judgement."
    )
    summary: str = Field(
        description="Concise explanation of the judgement referencing the available evidence."
    )


class ClaimFactCheckReport(BaseModel):
    """Structured report ready to present to end users for a single claim."""

    claim_text: str
    claim_confidence: float
    claim_category: str
    overall_stance: EvidenceStance
    verdict_confidence: VerdictConfidence
    verdict_summary: str
    evidence_by_stance: Dict[EvidenceStance, List[EvidenceSourceSummary]] = Field(
        default_factory=dict,
        description="Evidence grouped by stance for UI consumption.",
    )
    total_sources: int = Field(
        ge=0,
        description="Total evidence sources considered across all stances.",
    )
    timestamp_hint: float | None = Field(
        default=None,
        description="Suggested video timestamp in seconds where this claim appears",
    )
    timestamp_confidence: float | None = Field(
        default=None,
        description="Confidence score (0.0-1.0) for the timestamp hint",
    )


class FactCheckRunOutput(BaseModel):
    """Container aggregating extracted claims with their fact-check reports."""

    extracted_claims: ExtractedClaims
    claim_reports: List[ClaimFactCheckReport]
    transcript_data: TranscriptData = Field(
        description="Original transcript with timestamped segments for UI"
    )


class ClaimEvidenceBundle(BaseModel):
    """Helper payload that bundles the raw claim with associated search results."""

    claim: Claim
    stance_groups: Dict[EvidenceStance, List[EvidenceSourceSummary]]

    @property
    def total_sources(self) -> int:
        """Count how many evidence sources are available across stances."""
        return sum(len(items) for items in self.stance_groups.values())
