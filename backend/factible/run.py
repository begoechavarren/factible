import logging
from typing import Callable, List, Optional, Sequence, Tuple

from dotenv import load_dotenv

from factible.components.claim_extractor.claim_extractor import extract_claims
from factible.components.claim_extractor.schemas import Claim, ExtractedClaims
from factible.components.online_search.schemas.search import SearchResult
from factible.components.online_search.search import search_online
from factible.components.output_generator.output_generator import generate_run_output
from factible.components.output_generator.schemas import FactCheckRunOutput
from factible.components.query_generator.query_generator import generate_queries
from factible.components.transcriptor.transcriptor import get_transcript
from factible.utils.profile import timer

_logger = logging.getLogger(__name__)


# TODO: Move logging away from run_factible orchestrator function
def run_factible(
    video_url: str,
    *,
    max_claims: Optional[int] = 5,
    enable_search: bool = True,
    max_queries_per_claim: int = 2,
    max_results_per_query: int = 3,
    headless_search: bool = True,
    progress_callback: Optional[Callable[[str, str, int, dict], None]] = None,
) -> FactCheckRunOutput:
    """
    Run the factible agent.

    Args:
        video_url: The URL of the YouTube video to fact check.
        max_claims: Optional cap on how many top-importance claims are processed.
        enable_search: Toggle the search stage to save time or API calls.
        max_queries_per_claim: Number of high-priority queries to run per claim.
        max_results_per_query: Number of search results to inspect per query.
        headless_search: Whether Selenium should run in headless mode when scraping.
        progress_callback: Optional callback for progress updates.
            Called with (step, message, progress, data).

    Returns:
        Structured fact-check reports for the processed claims.
    """
    with timer("TOTAL PIPELINE"):
        # Step 1: Extract transcript from YouTube video
        if progress_callback:
            progress_callback(
                "transcript_extraction",
                "Extracting transcript from YouTube video...",
                5,
                {},
            )

        with timer("Step 1: Transcript extraction"):
            transcript_text = get_transcript(video_url)

        if not transcript_text.strip():
            _logger.warning("No transcript retrieved for %s", video_url)
            if progress_callback:
                progress_callback(
                    "error",
                    "No transcript found for this video",
                    100,
                    {"error": "No transcript available"},
                )
            empty_claims = ExtractedClaims(claims=[], total_count=0)
            return generate_run_output(empty_claims, [])

        if progress_callback:
            progress_callback(
                "transcript_complete",
                f"Transcript extracted ({len(transcript_text)} characters)",
                15,
                {"transcript_length": len(transcript_text)},
            )

        # Step 2: Extract claims from transcript
        if progress_callback:
            progress_callback(
                "claim_extraction",
                "Extracting factual claims from transcript...",
                20,
                {},
            )

        with timer("Step 2: Claim extraction"):
            extracted_claims = extract_claims(transcript_text, max_claims=max_claims)

        total_claims = extracted_claims.total_count
        claims_to_process = list(extracted_claims.claims)
        if 0 <= (max_claims or -1) and len(claims_to_process) < total_claims:
            _logger.info(
                "Processing limited to top %d of %d claims by importance",
                len(claims_to_process),
                total_claims,
            )

        # Show extracted claims
        _logger.info(f"=== EXTRACTED CLAIMS ({total_claims}) ===")
        for i, claim in enumerate(claims_to_process, 1):
            _logger.info(
                "Claim %d: [%s] (confidence: %.2f | importance: %.2f)",
                i,
                claim.category,
                claim.confidence,
                claim.importance,
            )
            _logger.info(f"  {claim.text}")
            if claim.context:
                _logger.info(f"  Context: {claim.context}")

        if progress_callback:
            progress_callback(
                "claims_extracted",
                f"Extracted {total_claims} claims",
                35,
                {
                    "total_claims": total_claims,
                    "processing_claims": len(claims_to_process),
                    "claims": [claim.model_dump() for claim in claims_to_process],
                },
            )

        processed_claims = ExtractedClaims(
            claims=claims_to_process,
            total_count=len(claims_to_process),
        )

        claim_evidence_records: List[Tuple[Claim, Sequence[SearchResult]]] = []

        if not claims_to_process:
            _logger.info("No claims available for search; pipeline complete.")
            if progress_callback:
                progress_callback(
                    "complete",
                    "No claims to fact-check",
                    100,
                    {"result": processed_claims.model_dump()},
                )
            return generate_run_output(processed_claims, [])

        if not enable_search:
            _logger.info("Search disabled; skipping fact-checking stage.")
            for claim in claims_to_process:
                claim_evidence_records.append((claim, []))
            return generate_run_output(processed_claims, claim_evidence_records)

        # Step 3: Generate search queries and search for each claim
        base_progress = 35
        progress_per_claim = 50 // max(1, len(claims_to_process))

        def _process_claim(
            index: int, claim: Claim
        ) -> Tuple[Claim, List[SearchResult]]:
            _logger.info(f"--- SEARCH RESULTS FOR CLAIM {index} ---")
            collected_results: List[SearchResult] = []

            if max_queries_per_claim <= 0:
                _logger.info("  Query execution skipped (max_queries_per_claim <= 0)")
                return claim, collected_results

            try:
                with timer(f"  Query generation for claim {index}"):
                    queries = generate_queries(
                        claim,
                        max_queries=max_queries_per_claim,
                        priority_threshold=2,
                    )
            except Exception as exc:  # pragma: no cover - defensive path
                _logger.error("Failed to generate queries for claim %d: %s", index, exc)
                return claim, collected_results

            if not queries.queries:
                _logger.info("  No high-priority queries generated for this claim")
                return claim, collected_results

            for query_index, query_obj in enumerate(queries.queries, 1):
                _logger.info(f"Query {query_index}: {query_obj.query}")
                if max_results_per_query <= 0:
                    _logger.info("  Search skipped (max_results_per_query <= 0)")
                    continue

                try:
                    with timer(f"    Search execution for query {query_index}"):
                        search_results = search_online(
                            query_obj.query,
                            limit=max(1, max_results_per_query),
                            headless=headless_search,
                        )
                except Exception as exc:  # pragma: no cover - defensive path
                    _logger.error("  Search failed: %s", exc)
                    continue

                if search_results.total_count == 0:
                    _logger.info("  No search results found")
                    continue

                collected_results.extend(search_results.results)

                for k, result in enumerate(search_results.results, 1):
                    reliability = result.reliability
                    reliability_label = reliability.rating.upper()
                    _logger.info(
                        "  %d. %s [%s | %.2f]",
                        k,
                        result.title,
                        reliability_label,
                        reliability.score,
                    )
                    _logger.info(f"     URL: {result.url}")

                    if result.evidence_summary:
                        _logger.info(
                            f"     Evidence summary: {result.evidence_summary}"
                        )

                    if result.evidence_overall_stance:
                        _logger.info(
                            "     Evidence stance: %s",
                            result.evidence_overall_stance.upper(),
                        )

            return claim, collected_results

        if claims_to_process:
            with timer("Step 3: Query generation + search for all claims"):
                for idx, claim in enumerate(claims_to_process, 1):
                    current_progress = base_progress + (idx * progress_per_claim)
                    if progress_callback:
                        progress_callback(
                            f"processing_claim_{idx}",
                            f"Processing claim {idx}/{len(claims_to_process)}: {claim.text[:50]}...",
                            current_progress,
                            {"claim_index": idx, "claim_text": claim.text},
                        )
                    claim_evidence_records.append(_process_claim(idx, claim))

        if progress_callback:
            progress_callback(
                "generating_report",
                "Generating final fact-check report...",
                90,
                {},
            )

        with timer("Step 4: Output generation"):
            run_output = generate_run_output(processed_claims, claim_evidence_records)

        for idx, report in enumerate(run_output.claim_reports, 1):
            _logger.info(f"=== FACT-CHECK REPORT FOR CLAIM {idx} ===")
            _logger.info(
                "Overall stance: %s (confidence: %s)",
                report.overall_stance.upper(),
                report.verdict_confidence,
            )
            _logger.info(f"Summary: {report.verdict_summary}")
            for stance, sources in report.evidence_by_stance.items():
                _logger.info(f"  {stance.upper()} ({len(sources)} sources)")
                for source in sources:
                    reliability = source.reliability
                    _logger.info(
                        "    - %s [%s | %.2f]",
                        source.title,
                        reliability.rating.upper(),
                        reliability.score,
                    )
                    _logger.info(f"      URL: {source.url}")
                    if source.evidence_summary:
                        _logger.info(f"      Evidence: {source.evidence_summary}")

        if progress_callback:
            progress_callback(
                "complete",
                "Fact-checking complete!",
                100,
                {"result": run_output.model_dump()},
            )

        return run_output


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    VIDEO_URL = "https://www.youtube.com/watch?v=iGkLcqLWxMA"

    result = run_factible(
        video_url=VIDEO_URL,
        max_claims=1,
        max_queries_per_claim=1,  # TODO: Remove
        max_results_per_query=1,  # TODO: Remove
    )
    _logger.info("Completed processing %d claims.", result.extracted_claims.total_count)
