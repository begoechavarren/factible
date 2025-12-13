import asyncio
import logging
from typing import Callable, List, Optional

from dotenv import load_dotenv

from factible.components.claim_extractor.claim_extractor import extract_claims
from factible.components.claim_extractor.schemas import Claim, ExtractedClaims
from factible.components.online_search.schemas.search import SearchResult, SearchResults
from factible.components.online_search.search import search_online_async
from factible.components.output_generator.output_generator import (
    generate_claim_report,
    generate_run_output,
)
from factible.components.output_generator.schemas import (
    ClaimFactCheckReport,
    FactCheckRunOutput,
)
from factible.components.query_generator.query_generator import generate_queries
from factible.components.transcriptor.transcriptor import (
    get_transcript_with_segments,
    get_video_title,
)
from factible.models.config import (
    CLAIM_EXTRACTOR_MODEL,
    EVIDENCE_EXTRACTOR_MODEL,
    OUTPUT_GENERATOR_MODEL,
    QUERY_GENERATOR_MODEL,
)
from factible.tracking.tracker import ExperimentTracker, timer

_logger = logging.getLogger(__name__)


async def run_factible(
    video_url: str,
    *,
    experiment_name: str = "default",
    max_claims: Optional[int] = 5,
    max_queries_per_claim: int = 2,
    max_results_per_query: int = 3,
    min_source_credibility: str = "medium",
    headless_search: bool = True,
    progress_callback: Optional[Callable[[str, str, int, dict], None]] = None,
    runs_subdir: Optional[str] = None,
    run_label: Optional[str] = None,
) -> FactCheckRunOutput:
    """
    Async parallelization at 3 levels:
    - Level 1: Process claims in parallel
    - Level 2: Process queries per claim in parallel
    - Level 3: Process search results per query in parallel
    """
    model_config = {
        "claim_extractor": CLAIM_EXTRACTOR_MODEL.name,
        "query_generator": QUERY_GENERATOR_MODEL.name,
        "evidence_extractor": EVIDENCE_EXTRACTOR_MODEL.name,
        "output_generator": OUTPUT_GENERATOR_MODEL.name,
    }
    model_summary = " | ".join(
        [
            f"CE:{CLAIM_EXTRACTOR_MODEL.value.model_name}",
            f"QG:{QUERY_GENERATOR_MODEL.value.model_name}",
            f"EE:{EVIDENCE_EXTRACTOR_MODEL.value.model_name}",
            f"OG:{OUTPUT_GENERATOR_MODEL.value.model_name}",
        ]
    )

    config = {
        "video_url": video_url,
        "max_claims": max_claims,
        "max_queries_per_claim": max_queries_per_claim,
        "max_results_per_query": max_results_per_query,
        "headless_search": headless_search,
        "model_config": model_config,
        "model_summary": model_summary,
        "parallelization": "async",  # Mark as async version
    }

    from pathlib import Path

    base_dir = Path("factible/experiments/runs")
    if runs_subdir:
        base_dir = base_dir / runs_subdir

    with ExperimentTracker(
        "end_to_end",
        experiment_name,
        config,
        base_dir=base_dir,
        run_label=run_label,
    ) as tracker:
        tracker.log_input("video_url", video_url)

        # Fetch video title
        video_title = get_video_title(video_url)
        tracker.log_input("video_title", video_title)

        # Step 1: Extract transcript (not parallelizable)
        if progress_callback:
            progress_callback(
                "transcript_extraction",
                "Extracting transcript from YouTube video...",
                5,
                {},
            )

        with timer("Step 1: Transcript extraction"):
            transcript_data = get_transcript_with_segments(video_url)
            transcript_text = transcript_data.text

        tracker.log_input("transcript_length", len(transcript_text))
        tracker.log_input("transcript_tokens", len(transcript_text.split()))

        if transcript_data.segments:
            last_segment = transcript_data.segments[-1]
            video_duration_seconds = last_segment.start + last_segment.duration
            tracker.log_input("video_duration_seconds", video_duration_seconds)

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
            return await generate_run_output(empty_claims, [], transcript_data)

        if progress_callback:
            progress_callback(
                "transcript_complete",
                f"Transcript extracted ({len(transcript_text)} characters)",
                15,
                {"transcript_length": len(transcript_text)},
            )

        # Step 2: Extract claims (not parallelizable - single LLM call)
        if progress_callback:
            progress_callback(
                "claim_extraction",
                "Extracting factual claims from transcript...",
                20,
                {},
            )

        with timer("Step 2: Claim extraction"):
            extracted_claims = await extract_claims(
                transcript_text, max_claims=max_claims
            )

        tracker.log_output("extracted_claims", extracted_claims.model_dump())

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

        if not claims_to_process:
            _logger.info("No claims available for search; pipeline complete.")
            if progress_callback:
                progress_callback(
                    "complete",
                    "No claims to fact-check",
                    100,
                    {"result": processed_claims.model_dump()},
                )
            return await generate_run_output(processed_claims, [], transcript_data)

        # Step 3: Process all claims in PARALLEL (evidence + verdict generation)
        async def _process_claim_async(
            index: int, claim: Claim
        ) -> ClaimFactCheckReport:
            """Process a single claim asynchronously: collect evidence and generate verdict (Level 1 parallelization)."""
            tracker.set_context(claim_index=index, claim_text=claim.text)

            _logger.info(f"--- SEARCH RESULTS FOR CLAIM {index} ---")
            collected_results: List[SearchResult] = []

            if max_queries_per_claim <= 0:
                _logger.info("  Query execution skipped (max_queries_per_claim <= 0)")
                # Generate verdict with no evidence
                report = await generate_claim_report(
                    claim, collected_results, transcript_data
                )
                tracker.clear_context()
                return report

            try:
                with timer(f"Step 3.{index}.1: Query generation for claim {index}"):
                    queries = await generate_queries(
                        claim,
                        max_queries=max_queries_per_claim,
                        priority_threshold=2,
                    )
            except Exception as exc:
                _logger.error("Failed to generate queries for claim %d: %s", index, exc)
                # Generate verdict with no evidence
                report = await generate_claim_report(
                    claim, collected_results, transcript_data
                )
                tracker.clear_context()
                return report

            if not queries.queries:
                _logger.info("  No high-priority queries generated for this claim")
                # Generate verdict with no evidence
                report = await generate_claim_report(
                    claim, collected_results, transcript_data
                )
                tracker.clear_context()
                return report

            # Level 2: Execute all queries in PARALLEL
            async def _execute_query(query_index: int, query_obj):
                tracker.set_context(
                    claim_index=index,
                    claim_text=claim.text,
                    query_index=query_index,
                    query_text=query_obj.query,
                )

                _logger.info(f"Query {query_index}: {query_obj.query}")
                if max_results_per_query <= 0:
                    _logger.info("  Search skipped (max_results_per_query <= 0)")
                    return SearchResults(
                        query=query_obj.query, results=[], total_count=0
                    )

                try:
                    with timer(
                        f"Step 3.{index}.2: Search execution for claim {index} query {query_index}"
                    ):
                        search_results = await search_online_async(
                            claim.text,
                            query_obj.query,
                            limit=max(1, max_results_per_query),
                            headless=headless_search,
                            min_credibility=min_source_credibility,
                            claim_index=index,
                            query_index=query_index,
                        )
                    return search_results
                except Exception as exc:
                    _logger.error("  Search failed: %s", exc)
                    return SearchResults(
                        query=query_obj.query, results=[], total_count=0
                    )

            # Execute all queries in parallel
            _logger.info(f"  ⚡ Executing {len(queries.queries)} queries in PARALLEL")
            query_tasks = [
                _execute_query(query_index, query_obj)
                for query_index, query_obj in enumerate(queries.queries, 1)
            ]
            search_results_list = await asyncio.gather(*query_tasks)

            # Flatten and collect results with deduplication by URL
            seen_urls: set[str] = set()
            duplicate_count = 0
            for search_results in search_results_list:
                if search_results.total_count > 0:
                    # First pass: deduplicate and collect
                    unique_results = []
                    for result in search_results.results:
                        # Deduplicate by URL to avoid duplicate sources across queries
                        if result.url in seen_urls:
                            duplicate_count += 1
                            _logger.debug(
                                "Skipping duplicate URL: %s (%s)",
                                result.url,
                                result.title,
                            )
                            continue
                        seen_urls.add(result.url)
                        collected_results.append(result)
                        unique_results.append(result)

                    # Second pass: log only unique results
                    for k, result in enumerate(unique_results, 1):
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

            if duplicate_count > 0:
                _logger.info(
                    f"Filtered {duplicate_count} duplicate URL(s) for claim {index}"
                )

            # Generate verdict immediately after evidence collection
            _logger.info(f"Generating verdict for claim {index}...")
            with timer(f"Step 3.{index}.3: Verdict generation for claim {index}"):
                report = await generate_claim_report(
                    claim, collected_results, transcript_data
                )

            tracker.clear_context()
            return report

        # Level 1: Process all claims in parallel (including verdict generation)
        _logger.info(f"⚡ Processing {len(claims_to_process)} claims in PARALLEL")
        with timer(
            "Step 3: Query generation + search + verdicts for all claims (PARALLEL)"
        ):
            claim_tasks = [
                _process_claim_async(idx, claim)
                for idx, claim in enumerate(claims_to_process, 1)
            ]
            claim_reports = await asyncio.gather(*claim_tasks)
        _logger.info(f"✓ Completed {len(claims_to_process)} claims in parallel")

        if progress_callback:
            progress_callback(
                "generating_report",
                "Sorting and finalizing fact-check report...",
                90,
                {},
            )

        # Step 4: Sort reports by evidence quality and assemble final output
        with timer("Step 4: Sort and assemble final output"):
            sorted_reports = sorted(
                claim_reports, key=lambda r: r.evidence_quality_score, reverse=True
            )
            _logger.info("✓ Sorted %d reports by evidence quality", len(sorted_reports))

            run_output = FactCheckRunOutput(
                extracted_claims=processed_claims,
                claim_reports=sorted_reports,
                transcript_data=transcript_data,
            )

        tracker.log_output("final_output", run_output.model_dump())

        # Log results
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

    VIDEO_URL = "https://www.youtube.com/watch?v=KdMqCUa7XJk"

    import time

    start_time = time.time()

    result = asyncio.run(
        run_factible(
            video_url=VIDEO_URL,
            experiment_name="async_test",
            max_claims=2,
            max_queries_per_claim=2,
            max_results_per_query=2,
            headless_search=True,
        )
    )

    elapsed = time.time() - start_time

    _logger.info("=" * 80)
    _logger.info(f"Async pipeline {elapsed:.2f} seconds")
    _logger.info(f"Processed {result.extracted_claims.total_count} claims")
    _logger.info(f"Generated {len(result.claim_reports)} reports")
    _logger.info("=" * 80)
