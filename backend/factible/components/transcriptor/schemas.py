from pydantic import BaseModel, Field


class TranscriptSegment(BaseModel):
    """A single timestamped segment from a YouTube transcript."""

    text: str = Field(description="The text content of this segment")
    start: float = Field(ge=0.0, description="Start time in seconds")
    duration: float = Field(ge=0.0, description="Duration in seconds")


class TranscriptData(BaseModel):
    """Complete transcript data with both plain text and timestamped segments."""

    text: str = Field(description="Full transcript as plain text")
    segments: list[TranscriptSegment] = Field(
        description="Individual timestamped segments"
    )
    video_id: str = Field(description="YouTube video ID")
