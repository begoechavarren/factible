import logging
import re
from difflib import SequenceMatcher

from pydantic_ai import Agent
from pydantic_ai.exceptions import AgentRunError

from factible.components.claim_extractor.schemas import ExtractedClaims
from factible.evaluation.pydantic_monitor import track_pydantic
from factible.models.config import CLAIM_EXTRACTOR_MODEL
from factible.models.llm import get_model

_logger = logging.getLogger(__name__)


def _normalize_for_matching(text: str) -> str:
    """Normalize text for fuzzy matching by removing punctuation and extra spaces."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _find_claim_in_transcript(
    claim_text: str, transcript: str, min_score: float = 0.5
) -> dict | None:
    """
    Find the best matching location for a claim in the original transcript.
    Uses sliding window with fuzzy string matching.

    Returns:
        dict with char_start, char_end, score, excerpt if found, else None
    """
    claim_normalized = _normalize_for_matching(claim_text)
    transcript_normalized = _normalize_for_matching(transcript)

    claim_words = claim_normalized.split()
    transcript_words = transcript_normalized.split()

    if not claim_words or not transcript_words:
        return None

    window_size = len(claim_words)
    best_match: dict[str, int | float | str] | None = None
    best_ratio = 0.0

    # Try different window sizes (exact, +/-2 words) to handle slight length variations
    for size_offset in [0, 1, 2, -1, -2]:
        current_window_size = max(3, window_size + size_offset)

        for i in range(len(transcript_words) - current_window_size + 1):
            window_words = transcript_words[i : i + current_window_size]
            window_text = " ".join(window_words)

            ratio = SequenceMatcher(None, claim_normalized, window_text).ratio()

            if ratio > best_ratio:
                best_ratio = ratio
                # Find the character positions in the original transcript
                # by mapping word positions back to character positions
                words_before = " ".join(transcript_words[:i])
                char_start_approx = len(words_before) + (1 if words_before else 0)

                # Find actual character position in original (case-insensitive)
                search_pattern = re.escape(" ".join(window_words[:3]))
                match = re.search(
                    search_pattern, transcript_normalized[char_start_approx:]
                )

                if match:
                    actual_start = char_start_approx + match.start()
                    actual_end = actual_start + len(window_text)

                    best_match = {
                        "char_start": actual_start,
                        "char_end": actual_end,
                        "score": ratio,
                        "excerpt": transcript[actual_start:actual_end],
                    }

    if best_match and float(best_match["score"]) >= min_score:
        _logger.info(
            "Found claim in transcript (score: %.2f): %s",
            best_match["score"],
            claim_text[:50],
        )
        return best_match

    _logger.warning(
        "Could not find claim in transcript (best score: %.2f): %s",
        best_ratio,
        claim_text[:50],
    )
    return None


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
        - context: Short note that captures timeframe, speaker, or situational details needed to fact-check the claim (e.g., "2016 US presidential debate", "speaker: Donald Trump"). When possible, also mention how it connects to the inferred thesis. If unclear, state "context unknown".

        Before listing claims, infer the video's central thesis in no more than 25 words
        (e.g., "Climate change alarmism is driven more by politics and media than by settled science").
        Use this thesis to judge how critical each claim is.

        When assigning IMPORTANCE scores:
        - 0.85-1.0 → Prescriptive or causal claims that, if false, would undermine the thesis (e.g., who/what is to blame, proposed solutions).
        - 0.60-0.80 → Quantitative or historical evidence directly tied to the thesis.
        - 0.30-0.55 → Context or supporting background needed to understand the story but not decisive.
        - 0.0-0.25 → Peripheral or anecdotal details that do not meaningfully change the argument.

        Workflow:
        1. Brainstorm candidate claims from the transcript.
        2. Rank them by thesis impact first, then uniqueness (remove paraphrases or repeated numbers).
        3. Only output the highest-ranked set (aim for the 8-10 most check-worthy items unless the transcript has fewer). Adjust importance scores slightly to reflect the ranking—avoid identical scores unless truly equivalent.

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


@track_pydantic("claim_extraction")
async def extract_claims(
    transcript: str, *, max_claims: int | None = None
) -> ExtractedClaims:
    """Extract factual claims from a transcript (async)."""
    agent = _get_claim_extractor_agent()
    try:
        result = await agent.run(
            f"Extract all factual claims from this YouTube transcript:\n\n{transcript}"
        )
    except AgentRunError as exc:
        _logger.error("Claim extraction failed: %s", exc)
        return ExtractedClaims(claims=[], total_count=0)
    extracted = result.output

    # Post-processing: Find each claim's position in the original transcript
    _logger.info("Post-processing: Finding claim positions in transcript...")
    for claim in extracted.claims:
        match_info = _find_claim_in_transcript(claim.text, transcript)
        if match_info:
            claim.transcript_char_start = match_info["char_start"]
            claim.transcript_char_end = match_info["char_end"]
            claim.transcript_match_score = match_info["score"]

    sorted_claims = sorted(
        extracted.claims,
        key=lambda claim: claim.importance,
        reverse=True,
    )

    if max_claims is not None and max_claims >= 0:
        sorted_claims = sorted_claims[:max_claims]

    return ExtractedClaims(claims=sorted_claims, total_count=len(sorted_claims))
