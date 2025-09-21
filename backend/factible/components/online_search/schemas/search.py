from typing import List

from pydantic import BaseModel, Field

from factible.components.online_search.schemas.evidence import (
    EvidenceSnippet,
    EvidenceStance,
)
from factible.components.online_search.schemas.reliability import SiteReliability


class SearchResult(BaseModel):
    """Enriched search result returned by the pipeline."""

    title: str
    url: str
    snippet: str
    engine: str
    reliability: SiteReliability
    relevant_evidence: List[EvidenceSnippet] = Field(default_factory=list)
    evidence_summary: str | None = None
    evidence_overall_stance: EvidenceStance | None = None
    content_characters: int = 0


class SearchResults(BaseModel):
    """Top-level container for search results."""

    query: str
    results: List[SearchResult]
    total_count: int
