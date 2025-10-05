import logging

from pydantic_ai import Agent
from pydantic_ai.exceptions import AgentRunError

from factible.components.claim_extractor.schemas import ExtractedClaims
from factible.models.config import CLAIM_EXTRACTOR_MODEL
from factible.models.llm import get_model

_logger = logging.getLogger(__name__)


def _get_claim_extractor_agent() -> Agent:
    """Get the claim extractor agent instance."""
    return Agent(
        model=get_model(CLAIM_EXTRACTOR_MODEL),
        output_type=ExtractedClaims,  # type: ignore[arg-type]
        retries=3,
        system_prompt="""
        You are an expert fact-checker. Your task is to extract factual claims from YouTube video transcripts.

        A factual claim is a statement that:
        1. Makes a specific assertion about reality
        2. Can potentially be verified or fact-checked
        3. Is not just an opinion, belief, or subjective statement

        For each claim you identify provide the following fields:
        - text: Extract the exact statement or a precise paraphrase
        - confidence: Confidence score (0.0-1.0) that this is a factual, checkable claim
        - category: Topic label (historical, scientific, statistical, biographical, geographical, policy, etc.)
        - importance: Score (0.0-1.0) capturing how impactful or controversial the claim is for fact-checking. Higher = more urgent to verify.
        - context: Short note that captures timeframe, speaker, or situational details needed to fact-check the claim (e.g., "2016 US presidential debate", "speaker: Donald Trump"). If unclear, state "context unknown".

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

        When judging importance, favor claims that:
        - Make strong, verifiable assertions that could influence beliefs or policy
        - Contain numbers, dates, or comparisons that could be misleading if false
        - Reference events or policies that have significant public impact

        Provide context that makes it possible to search for the claim (e.g., include year, event, or speaker if inferable). If the transcript implies the claim refers to past events, clarify that in the context note.

        Be thorough but precise. Quality over quantity.
        """,
    )


def extract_claims(
    transcript: str, *, max_claims: int | None = None
) -> ExtractedClaims:
    """Extract factual claims from a transcript."""
    agent = _get_claim_extractor_agent()
    try:
        result = agent.run_sync(
            f"Extract all factual claims from this YouTube transcript:\n\n{transcript}"
        )
    except AgentRunError as exc:
        _logger.error("Claim extraction failed: %s", exc)
        return ExtractedClaims(claims=[], total_count=0)
    extracted = result.output
    sorted_claims = sorted(
        extracted.claims,
        key=lambda claim: claim.importance,
        reverse=True,
    )

    if max_claims is not None and max_claims >= 0:
        sorted_claims = sorted_claims[:max_claims]

    return ExtractedClaims(claims=sorted_claims, total_count=len(extracted.claims))
