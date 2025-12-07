from pydantic import BaseModel, Field, HttpUrl


class FactCheckRequest(BaseModel):
    """Request schema for fact-checking a YouTube video."""

    video_url: HttpUrl = Field(
        ...,
        description="YouTube video URL to fact-check",
        examples=["https://www.youtube.com/watch?v=iGkLcqLWxMA"],
    )
    experiment_name: str = Field(
        default="default",
        description="Name for this experiment run (for tracking)",
    )
    max_claims: int | None = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of claims to extract and verify",
    )
    max_queries_per_claim: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Number of search queries to generate per claim",
    )
    max_results_per_query: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of search results to analyze per query",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "video_url": "https://www.youtube.com/watch?v=iGkLcqLWxMA",
                "max_claims": 5,
                "max_queries_per_claim": 2,
                "max_results_per_query": 3,
            }
        }
