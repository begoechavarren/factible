import asyncio
import logging
from typing import List, Sequence, Tuple

from pydantic_ai import Agent

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
from factible.evaluation.pydantic_monitor import track_pydantic
from factible.evaluation.tracker import ExperimentTracker
from factible.models.config import OUTPUT_GENERATOR_MODEL
from factible.models.llm import get_model

_logger = logging.getLogger(__name__)

# All possible stances in the evidence schema
_ALL_STANCES: Sequence[EvidenceStance] = ("supports", "refutes", "mixed", "unclear")

# Actionable stances for verdict generation (exclude "unclear" as it provides no verification value)
_ACTIONABLE_STANCES: Sequence[EvidenceStance] = ("supports", "refutes", "mixed")


def _build_claim_verdict_agent() -> Agent:
    """Instantiate the LLM agent responsible for producing claim verdicts."""
    return Agent(
        model=get_model(OUTPUT_GENERATOR_MODEL),
        output_type=ClaimVerdict,  # type: ignore[arg-type]
        system_prompt="""
        You are an expert fact-checking analyst. Your goal is to evaluate how the provided evidence
        supports, refutes, or provides mixed/unclear signals for a specific claim. Base every
        judgement strictly on the supplied evidence summary and reliability information.

        Output requirements:
        - Choose overall_stance as one of: supports, refutes, mixed, unclear.
        - Set confidence to low/medium/high depending on the strength, consistency, and reliability
          of the evidence.
        - Provide a concise summary referencing the most persuasive evidence (cite stance and
          reliability when relevant).

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
    # Only include actionable stances for verdict generation
    # "unclear" sources are topically relevant but don't help verification
    grouped: dict[EvidenceStance, List[EvidenceSourceSummary]] = {
        stance: [] for stance in _ACTIONABLE_STANCES
    }

    unclear_count = 0
    for result in search_results:
        stance: EvidenceStance = result.evidence_overall_stance or "unclear"

        # Filter out unclear sources as they don't contribute to verification
        if stance == "unclear":
            unclear_count += 1
            _logger.debug(
                "Filtering unclear source from verdict: %s (%s)",
                result.title,
                result.url,
            )
            continue

        summary = _select_evidence_description(result)
        grouped[stance].append(
            EvidenceSourceSummary(
                title=result.title,
                url=result.url,
                reliability=result.reliability,
                stance=stance,
                evidence_summary=summary,
                snippet=result.snippet or None,
            )
        )

    if unclear_count > 0:
        _logger.info(
            "Filtered %d unclear source(s) for claim: %s",
            unclear_count,
            claim.text[:50],
        )

    # Drop empty stances to keep payload compact for the UI.
    compact_grouped = {stance: items for stance, items in grouped.items() if items}

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
            summary="No evidence was found, so the claim remains unverified.",
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
            "Failed to generate verdict for claim '%s': %s", bundle.claim.text, exc
        )
        return ClaimVerdict(
            overall_stance="unclear",
            confidence="low",
            summary="Automatic verdict generation failed; manual review required.",
        )

    return result.output


async def generate_claim_report(
    claim: Claim,
    search_results: Sequence[SearchResult],
    transcript_data: TranscriptData,
) -> ClaimFactCheckReport:
    """Create a fact-check report for a single claim from search evidence (async)."""
    bundle = _build_evidence_bundle(claim, search_results)
    verdict = await _generate_verdict(bundle)

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

    return FactCheckRunOutput(
        extracted_claims=extracted_claims,
        claim_reports=list(reports),
        transcript_data=transcript_data,
    )
