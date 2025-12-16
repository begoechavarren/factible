import re
import os
import logging
import requests

from typing import Optional
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi

from factible.components.transcriptor.schemas import TranscriptData, TranscriptSegment

from youtube_transcript_api.proxies import WebshareProxyConfig

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


def get_video_title(url: str) -> str:
    """Get video title using YouTube oEmbed API.

    Args:
        url: YouTube video URL

    Returns:
        Video title string, or "Unknown" if fetch fails
    """
    try:
        oembed_url = f"https://www.youtube.com/oembed?url={url}&format=json"
        response = requests.get(oembed_url, timeout=5)
        response.raise_for_status()
        data = response.json()
        return data.get("title", "Unknown")
    except Exception as exc:
        _logger.warning(f"Failed to fetch video title: {exc}")
        return "Unknown"


def get_transcript_with_segments(
    url: str, languages: Optional[list[str]] = None
) -> TranscriptData:
    """Get transcript with both plain text and timestamped segments.

    Tries direct API first, falls back to Webshare proxy if rate limited.

    Args:
        url: YouTube video URL
        languages: List of preferred languages (default: ["en", "en-US"])

    Returns:
        TranscriptData containing plain text, segments, and video_id
    """
    if languages is None:
        languages = ["en", "en-US"]

    video_id = extract_video_id(url)

    # Try direct API first
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id, languages=languages)
        _logger.debug(f"Fetched transcript for {video_id} (direct)")

    except Exception as e:
        error_msg = str(e).lower()

        # Check if it it is throwing a rate limiting error
        if (
            "blocking" in error_msg
            or "ipblocked" in error_msg
            or "requestblocked" in error_msg
        ):
            _logger.warning(f"Rate limited for {video_id}, trying proxy fallback...")

            # Try with Webshare proxy
            if all(
                [
                    os.getenv("WEBSHARE_PROXY_USERNAME"),
                    os.getenv("WEBSHARE_PROXY_PASSWORD"),
                ]
            ):
                proxy_config = WebshareProxyConfig(
                    proxy_username=os.getenv("WEBSHARE_PROXY_USERNAME"),
                    proxy_password=os.getenv("WEBSHARE_PROXY_PASSWORD"),
                    filter_ip_locations=[os.getenv("WEBSHARE_PROXY_LOCATION", "es")],
                )
                ytt_api = YouTubeTranscriptApi(proxy_config=proxy_config)
                transcript = ytt_api.fetch(video_id, languages=languages)
                _logger.info(f"Fetched transcript for {video_id} (via proxy)")
            else:
                _logger.error("Proxy credentials not configured, cannot fallback")
                raise
        else:
            raise

    # Extract segments with timestamps
    segments = [
        TranscriptSegment(
            text=snippet.text, start=snippet.start, duration=snippet.duration
        )
        for snippet in transcript
    ]

    # Join and clean for plain text
    text = " ".join(snippet.text for snippet in transcript)
    text = re.sub(r"\s+", " ", text).strip()

    return TranscriptData(text=text, segments=segments, video_id=video_id)


def map_char_position_to_timestamp(
    char_position: int, transcript_data: TranscriptData
) -> dict | None:
    """
    Map a character position in the full transcript to a segment timestamp.

    Args:
        char_position: Character index in the full transcript text
        transcript_data: TranscriptData with segments

    Returns:
        dict with segment_index, start, duration if found, else None
    """
    if not transcript_data.segments:
        return None

    # Create character position map
    current_char = 0
    for idx, segment in enumerate(transcript_data.segments):
        segment_text = segment.text + " "
        segment_length = len(segment_text)

        if current_char <= char_position < current_char + segment_length:
            return {
                "segment_index": idx,
                "start": segment.start,
                "duration": segment.duration,
            }

        current_char += segment_length

    # If position is after all the segments, return the last
    if transcript_data.segments:
        last_segment = transcript_data.segments[-1]
        return {
            "segment_index": len(transcript_data.segments) - 1,
            "start": last_segment.start,
            "duration": last_segment.duration,
        }

    return None
