from typing import List

from pydantic import BaseModel, Field

from factible.components.online_search.schemas.evidence import EvidenceStance
from factible.components.online_search.schemas.reliability import SiteReliability


class SearchResult(BaseModel):
    """Enriched search result returned by the pipeline."""

    title: str
    url: str
    snippet: str
    engine: str
    reliability: SiteReliability
    evidence_summary: str | None = None
    evidence_overall_stance: EvidenceStance | None = None
    content_characters: int = 0
    content_source: str = Field(
        default="scraped",
        description="Source of content used for evidence extraction: 'scraped' or 'snippet_fallback'",
    )


class SearchResults(BaseModel):
    """Top-level container for search results."""

    query: str
    results: List[SearchResult]
    total_count: int
