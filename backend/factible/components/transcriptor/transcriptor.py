import re
import logging

from typing import Optional
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi

from factible.components.transcriptor.schemas import TranscriptData, TranscriptSegment

_logger = logging.getLogger(__name__)


def extract_video_id(url: str) -> str:
    """Extract video ID from YouTube URL."""
    parsed = urlparse(url)

    if parsed.hostname in ("youtu.be", "www.youtu.be"):
        return parsed.path[1:]

    if parsed.hostname in ("youtube.com", "www.youtube.com"):
        if parsed.path == "/watch":
            return parse_qs(parsed.query)["v"][0]
        if parsed.path.startswith("/embed/"):
            return parsed.path.split("/")[2]
        if parsed.path.startswith("/v/"):
            return parsed.path.split("/")[2]

    raise ValueError(f"Invalid YouTube URL: {url}")


def get_transcript_with_segments(
    url: str, languages: Optional[list[str]] = None
) -> TranscriptData:
    """Get transcript with both plain text and timestamped segments.

    Args:
        url: YouTube video URL
        languages: List of preferred languages (default: ["en", "en-US"])

    Returns:
        TranscriptData containing plain text, segments, and video_id
    """
    if languages is None:
        languages = ["en", "en-US"]

    video_id = extract_video_id(url)

    ytt_api = YouTubeTranscriptApi()
    transcript = ytt_api.fetch(video_id, languages=languages)

    # Extract segments with timestamps
    segments = [
        TranscriptSegment(
            text=snippet.text, start=snippet.start, duration=snippet.duration
        )
        for snippet in transcript
    ]

    # Join snippets and clean up newlines for plain text
    text = " ".join(snippet.text for snippet in transcript)
    text = re.sub(r"\s+", " ", text).strip()

    return TranscriptData(text=text, segments=segments, video_id=video_id)


def get_transcript(url: str, languages: Optional[list[str]] = None) -> str:
    """Get transcript for a YouTube video.

    Args:
        url: YouTube video URL
        languages: List of preferred languages (default: ["en", "en-US"])

    Returns:
        Transcript text as a single string
    """
    transcript_data = get_transcript_with_segments(url, languages)
    return transcript_data.text
