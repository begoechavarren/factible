import logging

from factible.agents.transcript.youtube_transcript import get_transcript

_logger = logging.getLogger(__name__)


def run_factible(video_url: str) -> str:
    """
    Run the factible agent.

    Args:
        video_url: The URL of the YouTube video to fact check.

    Returns:
        The fact checked transcript.
    """
    transcript_text = get_transcript(video_url)

    _logger.info(f"Transcript text: {transcript_text}")
    return transcript_text


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    VIDEO_URL = "https://www.youtube.com/watch?v=K7JZ6pUNADU"

    result = run_factible(video_url=VIDEO_URL)
    print(result)
