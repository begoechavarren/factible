import asyncio
import logging
from typing import List, Sequence, Tuple

from pydantic_ai import Agent
from pydantic_ai.models import ModelSettings

from factible.components.claim_extractor.schemas import Claim, ExtractedClaims
from factible.components.online_search.schemas.evidence import EvidenceStance
from factible.components.online_search.schemas.search import SearchResult
from factible.components.output_generator.schemas import (
    ClaimEvidenceBundle,
    ClaimFactCheckReport,
    ClaimVerdict,
    EvidenceSourceSummary,
    FactCheckRunOutput,
)
from factible.components.transcriptor.schemas import TranscriptData
from factible.components.transcriptor.transcriptor import map_char_position_to_timestamp
from factible.tracking.pydantic_monitor import track_pydantic
from factible.tracking.tracker import ExperimentTracker
from factible.models.config import OUTPUT_GENERATOR_MODEL
from factible.models.llm import get_model

_logger = logging.getLogger(__name__)

OUTPUT_GENERATOR_MODEL_SETTINGS: ModelSettings = {
    "temperature": 0.0,
    "max_tokens": 900,
}

# All possible stances in the evidence schema
_ALL_STANCES: Sequence[EvidenceStance] = ("supports", "refutes", "mixed", "unclear")


def _build_claim_verdict_agent() -> Agent:
    """Instantiate the LLM agent responsible for producing claim verdicts."""
    return Agent(
        model=get_model(OUTPUT_GENERATOR_MODEL),
        output_type=ClaimVerdict,  # type: ignore[arg-type]
        model_settings=OUTPUT_GENERATOR_MODEL_SETTINGS,
        system_prompt="""
        You are an expert fact-checking analyst. Your goal is to evaluate how the provided evidence
        supports, refutes, or provides mixed/unclear signals for a specific claim. Base every
        judgement strictly on the supplied evidence summary and reliability information.

        Output requirements:
        - Choose overall_stance as one of: supports, refutes, mixed, unclear.
        - Set confidence to low/medium/high depending on the strength, consistency, and reliability
          of the evidence.
        - Provide a concise summary referencing the most persuasive evidence. When citing a specific
          piece of evidence, name the source once (e.g., "Frontiers study", "Nature article",
          "Reuters report", or "WHO guidance"). For
          consensus statements, keep the summary concise without enumerating every source. Cite stance
          and reliability when relevant, and only mention additional source names when they clarify
          contrasting perspectives.
        - Only when evidence conflicts, explicitly name the key supporting and refuting sources so the
          reader understands where disagreement originates. If there is consensus, simply synthesize it.

        If there is no meaningful evidence, select 'unclear' with low confidence and explain the gap.
        """,
    )


def _select_evidence_description(result: SearchResult) -> str | None:
    """Choose the most informative short description for a search result."""
    if result.evidence_summary:
        return result.evidence_summary
    snippet = (result.snippet or "").strip()
    return snippet or None


def _build_evidence_bundle(
    claim: Claim, search_results: Sequence[SearchResult]
) -> ClaimEvidenceBundle:
    # Group evidence by stance (including unclear for transparency)
    grouped: dict[EvidenceStance, List[EvidenceSourceSummary]] = {}
    reliability_priority = {"high": 0, "medium": 1, "low": 2, "unknown": 3}
    stance_priority: Sequence[EvidenceStance] = (
        "refutes",
        "supports",
        "mixed",
        "unclear",
    )

    duplicate_count = 0
    seen_urls: set[str] = set()

    for result in search_results:
        stance: EvidenceStance = result.evidence_overall_stance or "unclear"

        # Deduplicate by URL (same source may appear from different queries)
        if result.url in seen_urls:
            duplicate_count += 1
            _logger.debug(
                f"Filtering duplicate URL from verdict: {result.title} ({result.url})"
            )
            continue
        seen_urls.add(result.url)

        summary = _select_evidence_description(result)
        snippet = result.snippet or None

        # Add to appropriate stance group (including unclear for transparency)
        if stance not in grouped:
            grouped[stance] = []

        grouped[stance].append(
            EvidenceSourceSummary(
                title=result.title,
                url=result.url,
                reliability=result.reliability,
                stance=stance,
                evidence_summary=summary,
                snippet=snippet,
            )
        )

    if duplicate_count > 0:
        _logger.info(
            f"Filtered {duplicate_count} duplicate URL(s) for claim: {claim.text[:50]}"
        )

    # Sort each stance group so higher reliability sources appear first.
    for sources in grouped.values():
        sources.sort(
            key=lambda source: (
                reliability_priority.get(
                    source.reliability.rating, reliability_priority["unknown"]
                ),
                -source.reliability.score,
                (source.title or "").lower(),
            )
        )

    # Drop empty stances but preserve a canonical stance ordering so refutes/supports
    # appear before mixed/unclear across all clients.
    compact_grouped: dict[EvidenceStance, List[EvidenceSourceSummary]] = {}
    for stance in stance_priority:
        items = grouped.get(stance)
        if items:
            compact_grouped[stance] = items

    # Include any other stance keys that might exist in fallback order.
    for stance, items in grouped.items():
        if stance not in compact_grouped and items:
            compact_grouped[stance] = items

    return ClaimEvidenceBundle(claim=claim, stance_groups=compact_grouped)


def _format_evidence_for_prompt(bundle: ClaimEvidenceBundle) -> str:
    """Create a compact textual description of the evidence groups for the LLM."""
    if not bundle.stance_groups:
        return "No evidence was retrieved for this claim."

    lines: List[str] = []
    for stance, sources in bundle.stance_groups.items():
        lines.append(f"{stance.upper()} ({len(sources)} sources):")
        for idx, source in enumerate(sources, start=1):
            reliability = source.reliability
            summary = source.evidence_summary or "No summary available."
            lines.append(
                f"  {idx}. {source.title} | reliability={reliability.rating} ({reliability.score:.2f})"
            )
            lines.append(f"     URL: {source.url}")
            lines.append(f"     Summary: {summary}")
    return "\n".join(lines)


@track_pydantic("verdict_generation")
async def _generate_verdict(bundle: ClaimEvidenceBundle) -> ClaimVerdict:
    """Ask the LLM to produce an overall verdict for the claim (async)."""
    if not bundle.stance_groups:
        return ClaimVerdict(
            overall_stance="unclear",
            confidence="low",
            summary="No reliable evidence sources were found for this claim. This may indicate the claim is difficult to verify, too specific, or references information not available in public sources.",
        )

    agent = _build_claim_verdict_agent()
    prompt = f"""
Claim: {bundle.claim.text}
Category: {bundle.claim.category}
Extraction confidence: {bundle.claim.confidence:.2f}

Evidence:
{_format_evidence_for_prompt(bundle)}

Provide the structured verdict.
"""

    try:
        result = await agent.run(prompt)
    except Exception as exc:  # pragma: no cover - defensive fallback
        _logger.error(
            "Failed to generate verdict for claim '%s': %s",
            bundle.claim.text,
            exc,
            exc_info=True,
        )
        # Include error type in summary for debugging
        error_type = type(exc).__name__
        error_msg = str(exc)[:100]  # Truncate long errors
        return ClaimVerdict(
            overall_stance="unclear",
            confidence="low",
            summary=f"Automatic verdict generation failed ({error_type}: {error_msg}); manual review required.",
        )

    return result.output


def _calculate_evidence_quality_score(bundle: ClaimEvidenceBundle) -> float:
    """Calculate quality score based on evidence quantity, reliability, and stance clarity."""
    if bundle.total_sources == 0:
        return 0.0

    # Base score from having evidence
    score = 0.3

    # Bonus for actionable stances (supports/refutes/mixed vs unclear)
    actionable_count = sum(
        len(sources)
        for stance, sources in bundle.stance_groups.items()
        if stance in ("supports", "refutes", "mixed")
    )
    if actionable_count > 0:
        score += 0.3 * min(1.0, actionable_count / 3.0)  # Up to 3 sources

    # Bonus for high reliability sources
    high_reliability_count = sum(
        1
        for sources in bundle.stance_groups.values()
        for source in sources
        if source.reliability.rating in ("high", "medium")
    )
    if high_reliability_count > 0:
        score += 0.4 * min(1.0, high_reliability_count / 3.0)  # Up to 3 high-quality

    return min(1.0, score)


async def generate_claim_report(
    claim: Claim,
    search_results: Sequence[SearchResult],
    transcript_data: TranscriptData,
) -> ClaimFactCheckReport:
    """Create a fact-check report for a single claim from search evidence (async)."""
    bundle = _build_evidence_bundle(claim, search_results)
    verdict = await _generate_verdict(bundle)

    # Calculate evidence quality for sorting
    quality_score = _calculate_evidence_quality_score(bundle)

    # Map claim's character position to timestamp if available
    timestamp_hint = None
    timestamp_confidence = None
    if claim.transcript_char_start is not None:
        timestamp_info = map_char_position_to_timestamp(
            claim.transcript_char_start, transcript_data
        )
        if timestamp_info:
            timestamp_hint = timestamp_info["start"]
            timestamp_confidence = claim.transcript_match_score

    return ClaimFactCheckReport(
        claim_text=claim.text,
        claim_confidence=claim.confidence,
        claim_category=claim.category,
        overall_stance=verdict.overall_stance,
        verdict_confidence=verdict.confidence,
        verdict_summary=verdict.summary,
        evidence_by_stance=bundle.stance_groups,
        total_sources=bundle.total_sources,
        evidence_quality_score=quality_score,
        timestamp_hint=timestamp_hint,
        timestamp_confidence=timestamp_confidence,
    )


async def generate_run_output(
    extracted_claims: ExtractedClaims,
    claim_results: Sequence[Tuple[Claim, Sequence[SearchResult]]],
    transcript_data: TranscriptData,
) -> FactCheckRunOutput:
    """Aggregate claim reports for an entire pipeline run (async with parallel verdicts)."""
    tracker = ExperimentTracker.get_current()

    async def _generate_report_with_context(
        idx: int, claim: Claim, results: Sequence[SearchResult]
    ) -> ClaimFactCheckReport:
        # Set context for verdict generation traceability
        if tracker:
            tracker.set_context(claim_index=idx, claim_text=claim.text)

        report = await generate_claim_report(claim, results, transcript_data)

        # Clear context after generating report
        if tracker:
            tracker.clear_context()

        return report

    # Generate all reports in parallel
    _logger.info(f"⚡ Generating {len(claim_results)} verdicts in PARALLEL")
    report_tasks = [
        _generate_report_with_context(idx, claim, results)
        for idx, (claim, results) in enumerate(claim_results, 1)
    ]
    reports = await asyncio.gather(*report_tasks)

    # Sort reports by evidence quality (high quality first, no evidence last)
    sorted_reports = sorted(
        reports, key=lambda r: r.evidence_quality_score, reverse=True
    )
    _logger.info("✓ Sorted %d reports by evidence quality", len(sorted_reports))

    return FactCheckRunOutput(
        extracted_claims=extracted_claims,
        claim_reports=sorted_reports,
        transcript_data=transcript_data,
    )
