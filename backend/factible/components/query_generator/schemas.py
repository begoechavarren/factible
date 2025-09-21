from typing import List

from pydantic import BaseModel, Field


class SearchQuery(BaseModel):
    """A search query for fact-checking a claim."""

    query: str = Field(description="The search query string")
    query_type: str = Field(
        description="Type of query: 'direct', 'alternative', 'source', 'context'",
    )
    priority: int = Field(
        ge=1,
        le=5,
        description="Priority level (1=highest, 5=lowest)",
    )


class GeneratedQueries(BaseModel):
    """Collection of search queries generated for claims."""

    original_claim: str
    queries: List[SearchQuery]
    total_count: int
