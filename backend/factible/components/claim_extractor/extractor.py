import logging

from pydantic_ai import Agent

from factible.components.claim_extractor.schemas import ExtractedClaims

_logger = logging.getLogger(__name__)


def _get_claim_extractor_agent() -> Agent:
    """Get the claim extractor agent instance."""
    return Agent(
        model="openai:gpt-4o-mini",
        output_type=ExtractedClaims,  # type: ignore[arg-type]
        system_prompt="""
        You are an expert fact-checker. Your task is to extract factual claims from YouTube video transcripts.

        A factual claim is a statement that:
        1. Makes a specific assertion about reality
        2. Can potentially be verified or fact-checked
        3. Is not just an opinion, belief, or subjective statement

        For each claim you identify:
        - Extract the exact text or a clear paraphrase
        - Assign a confidence score (0.0-1.0) based on how certain you are it's a factual claim
        - Categorize it (historical, scientific, statistical, biographical, geographical, etc.)

        Focus on extracting claims that are:
        - Specific facts, dates, numbers, or statistics
        - Historical events or biographical information
        - Scientific assertions
        - Geographic or demographic statements
        - Policy or legal claims

        Ignore:
        - Pure opinions ("I think...", "In my view...")
        - Subjective statements ("beautiful", "amazing")
        - Future predictions without factual basis
        - Rhetorical questions
        - General advice or recommendations

        Be thorough but precise. Quality over quantity.
        """,
    )


def extract_claims(transcript: str) -> ExtractedClaims:
    """Extract factual claims from a transcript."""
    agent = _get_claim_extractor_agent()
    result = agent.run_sync(
        f"Extract all factual claims from this YouTube transcript:\n\n{transcript}"
    )

    return result.output
