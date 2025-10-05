from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy"] = "healthy"
    version: str


class ProgressUpdate(BaseModel):
    """SSE progress update during fact-checking."""

    step: str = Field(..., description="Current step identifier")
    message: str = Field(..., description="Human-readable progress message")
    progress: int = Field(
        ..., ge=0, le=100, description="Completion percentage (0-100)"
    )
    data: dict | None = Field(default=None, description="Optional step-specific data")

    class Config:
        json_schema_extra = {
            "example": {
                "step": "transcript_extraction",
                "message": "Extracting transcript from YouTube video...",
                "progress": 10,
                "data": None,
            }
        }


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: str | None = Field(default=None, description="Additional error details")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid YouTube URL",
                "detail": "The provided URL is not a valid YouTube video URL",
            }
        }
