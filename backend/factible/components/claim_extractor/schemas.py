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


class ExtractedClaims(BaseModel):
    """Collection of claims extracted from a transcript."""

    claims: List[Claim]
    total_count: int
