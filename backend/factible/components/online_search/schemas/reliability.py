from typing import List, Literal

from pydantic import BaseModel, Field


class SiteReliability(BaseModel):
    """Assessment of a website's reliability."""

    rating: Literal["high", "medium", "low", "unknown"]
    score: float = Field(ge=0.0, le=1.0)
    reasons: List[str] = Field(default_factory=list)
