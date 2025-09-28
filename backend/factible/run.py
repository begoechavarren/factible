import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Sequence, Tuple

from dotenv import load_dotenv

from factible.components.claim_extractor.extractor import extract_claims
from factible.components.claim_extractor.schemas import Claim, ExtractedClaims
from factible.components.online_search.schemas.search import SearchResult
from factible.components.online_search.search import search_online
from factible.components.output_generator.output_generator import generate_run_output
from factible.components.output_generator.schemas import FactCheckRunOutput
from factible.components.query_generator.query_generator import generate_queries
from factible.components.transcriptor.transcriptor import get_transcript

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
    enable_claim_parallelism: bool = False,
    max_parallel_claims: Optional[int] = None,
    enable_query_parallelism: bool = False,
    max_parallel_queries: Optional[int] = None,
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
        enable_claim_parallelism: Process separate claims concurrently when True.
        max_parallel_claims: Optional cap on concurrent claim workers (default: min(4, claims)).
        enable_query_parallelism: Process individual queries within a claim concurrently when True.
        max_parallel_queries: Optional cap on concurrent query workers per claim (default: min(4, queries)).

    Returns:
        Structured fact-check reports for the processed claims.
    """
    # Step 1: Extract transcript from YouTube video
    transcript_text = get_transcript(video_url)
    if not transcript_text.strip():
        _logger.warning("No transcript retrieved for %s", video_url)
        empty_claims = ExtractedClaims(claims=[], total_count=0)
        return generate_run_output(empty_claims, [])

    # Step 2: Extract claims from transcript
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

    processed_claims = ExtractedClaims(
        claims=claims_to_process,
        total_count=len(claims_to_process),
    )

    claim_evidence_records: List[Tuple[Claim, Sequence[SearchResult]]] = []

    if not enable_search:
        _logger.info("Search disabled; skipping fact-checking stage.")
        for claim in claims_to_process:
            claim_evidence_records.append((claim, []))
        return generate_run_output(processed_claims, claim_evidence_records)

    if not claims_to_process:
        _logger.info("No claims available for search; pipeline complete.")
        return generate_run_output(processed_claims, [])

    # Step 3: Generate search queries and search for each claim
    def _process_claim(index: int, claim: Claim) -> Tuple[Claim, List[SearchResult]]:
        _logger.info(f"--- SEARCH RESULTS FOR CLAIM {index} ---")
        collected_results: List[SearchResult] = []

        if max_queries_per_claim <= 0:
            _logger.info("  Query execution skipped (max_queries_per_claim <= 0)")
            return claim, collected_results

        try:
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

        def _execute_query(
            query_index: int, query_obj
        ) -> Tuple[int, List[SearchResult]]:
            _logger.info(f"Query {query_index}: {query_obj.query}")
            local_results: List[SearchResult] = []
            if max_results_per_query <= 0:
                _logger.info("  Search skipped (max_results_per_query <= 0)")
                return query_index, local_results
            try:
                search_results = search_online(
                    query_obj.query,
                    limit=max(1, max_results_per_query),
                    headless=headless_search,
                )
            except Exception as exc:  # pragma: no cover - defensive path
                _logger.error("  Search failed: %s", exc)
                return query_index, local_results

            if search_results.total_count == 0:
                _logger.info("  No search results found")
                return query_index, local_results

            local_results.extend(search_results.results)

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

                # if result.snippet:
                #     _logger.info(f"     Snippet: {result.snippet}")

                # if reliability.reasons:
                #     _logger.info("     Reliability signals:")
                #     for reason in reliability.reasons[:3]:
                #         _logger.info(f"       - {reason}")
                #     if len(reliability.reasons) > 3:
                #         _logger.info("       - ...")

                if result.evidence_summary:
                    _logger.info(f"     Evidence summary: {result.evidence_summary}")

                if result.evidence_overall_stance:
                    _logger.info(
                        "     Evidence stance: %s",
                        result.evidence_overall_stance.upper(),
                    )

                # if result.relevant_evidence:
                #     _logger.info("     Evidence snippets:")
                #     for snippet in result.relevant_evidence:
                #         stance = snippet.stance.upper()
                #         _logger.info(f"       - ({stance}) {snippet.text}")
                #         if snippet.rationale:
                #             _logger.info(f"         Reason: {snippet.rationale}")

            return query_index, local_results

        indexed_queries = list(enumerate(queries.queries, 1))
        if enable_query_parallelism and len(indexed_queries) > 1:
            query_workers = min(
                len(indexed_queries),
                max_parallel_queries
                if max_parallel_queries and max_parallel_queries > 0
                else 4,
            )
            _logger.info(
                "  Parallelizing %d queries with up to %d workers",
                len(indexed_queries),
                query_workers,
            )
            with ThreadPoolExecutor(max_workers=query_workers) as query_executor:
                future_to_index = {
                    query_executor.submit(_execute_query, q_idx, query_obj): q_idx
                    for q_idx, query_obj in indexed_queries
                }
                ordered_query_results: dict[int, List[SearchResult]] = {}
                for future in as_completed(future_to_index):
                    q_index = future_to_index[future]
                    try:
                        q_idx, q_results = future.result()
                        ordered_query_results[q_idx] = q_results
                    except Exception as exc:  # pragma: no cover - defensive path
                        _logger.error("  Query %d failed: %s", q_index, exc)
                        ordered_query_results[q_index] = []
                for q_idx, _ in indexed_queries:
                    collected_results.extend(ordered_query_results.get(q_idx, []))
        else:
            if enable_query_parallelism and len(indexed_queries) <= 1:
                _logger.info("  Parallel query execution skipped (only one query)")
            elif not enable_query_parallelism:
                _logger.info("  Processing queries sequentially")
            for q_idx, query_obj in indexed_queries:
                _, q_results = _execute_query(q_idx, query_obj)
                collected_results.extend(q_results)

        return claim, collected_results

    if claims_to_process:
        if enable_claim_parallelism and len(claims_to_process) > 1:
            claim_workers = min(
                len(claims_to_process),
                max_parallel_claims
                if max_parallel_claims and max_parallel_claims > 0
                else 4,
            )
            _logger.info(
                "Parallelizing %d claims with up to %d workers",
                len(claims_to_process),
                claim_workers,
            )
            with ThreadPoolExecutor(max_workers=claim_workers) as executor:
                future_to_index = {
                    executor.submit(_process_claim, idx, claim): idx - 1
                    for idx, claim in enumerate(claims_to_process, 1)
                }
                ordered_results: list[tuple[Claim, List[SearchResult]]] = [
                    (claims_to_process[idx], [])
                    for idx in range(len(claims_to_process))
                ]
                for future in as_completed(future_to_index):
                    slot = future_to_index[future]
                    try:
                        ordered_results[slot] = future.result()
                    except Exception as exc:  # pragma: no cover - defensive path
                        _logger.error("Claim processing failed: %s", exc)
                claim_evidence_records.extend(ordered_results)
        else:
            if enable_claim_parallelism and len(claims_to_process) <= 1:
                _logger.info("Parallel claim execution skipped (only one claim)")
            elif not enable_claim_parallelism:
                _logger.info("Processing claims sequentially")
            for idx, claim in enumerate(claims_to_process, 1):
                claim_evidence_records.append(_process_claim(idx, claim))

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

    return run_output


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    VIDEO_URL = "https://www.youtube.com/watch?v=iGkLcqLWxMA"

    result = run_factible(
        video_url=VIDEO_URL,
        max_claims=1,
        enable_query_parallelism=True,
    )
    _logger.info("Completed processing %d claims.", result.extracted_claims.total_count)
