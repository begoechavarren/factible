from typing import List

from pydantic import BaseModel, Field


class Claim(BaseModel):
    """A single claim or fact extracted from text."""

    text: str
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence (0.0-1.0) that this is a factual claim",
    )
    category: str
    importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Relevance score (0.0-1.0) indicating how strongly the claim underpins or challenges "
            "the video's thesis—higher scores mean fact-checking it could change the core argument."
        ),
    )
    context: str | None = Field(
        default=None,
        description="Concise context including timeframe or speaker cues that help interpret the claim.",
    )
    transcript_char_start: int | None = Field(
        default=None,
        description="Character position in transcript where this claim likely appears (post-processing)",
    )
    transcript_char_end: int | None = Field(
        default=None,
        description="End character position in transcript for this claim (post-processing)",
    )
    transcript_match_score: float | None = Field(
        default=None,
        description="Fuzzy match confidence score (0.0-1.0) for transcript position",
    )


class ExtractedClaims(BaseModel):
    """Collection of claims extracted from a transcript."""

    claims: List[Claim]
    total_count: int
