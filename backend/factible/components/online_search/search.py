import asyncio
import logging
from typing import List, Optional

from factible.components.online_search.schemas.search import SearchResult, SearchResults
from factible.components.online_search.steps.content_fetcher import (
    SeleniumContentFetcher,
)
from factible.components.online_search.steps.evidence_extractor import (
    RelevantContentExtractor,
)
from factible.components.online_search.steps.google_search import (
    GoogleSearchClient,
    GoogleSearchHit,
)
from factible.components.online_search.steps.reliability import (
    WebsiteReliabilityChecker,
)
from factible.evaluation.tracker import timer

_logger = logging.getLogger(__name__)


async def _process_search_result_async(
    item: GoogleSearchHit,
    claim: str,
    query: str,
    fetcher: Optional[SeleniumContentFetcher],
    reliability_checker: WebsiteReliabilityChecker,
    evidence_extractor: RelevantContentExtractor,
    prefix: str,
    result_idx: int,
) -> SearchResult:
    """Process a single search result asynchronously."""

    # Reliability assessment (fast, can run in thread pool)
    with timer(f"{prefix}.2: Reliability assessment (result {result_idx})"):
        reliability = await asyncio.to_thread(reliability_checker.assess, item.url)

    # Content fetching (blocking Selenium, run in thread pool)
    page_text = ""
    if fetcher is not None:
        with timer(f"{prefix}.3: Content fetching (result {result_idx})"):
            page_text = await fetcher.fetch_text_async(item.url)

    # Evidence extraction (LLM call, async)
    # Always pass both scraped content AND Google snippet
    # If scraping failed, use snippet as minimal content
    evidence_summary = None
    overall_stance = None
    content_for_extraction = page_text if page_text else item.snippet

    if content_for_extraction:
        with timer(f"{prefix}.4: Evidence extraction (result {result_idx})"):
            evidence_output = await evidence_extractor.extract(
                claim,
                query,
                content_for_extraction,
                title=item.title,
                snippet=item.snippet
                if page_text
                else None,  # Don't duplicate if using snippet as content
            )
            overall_stance = evidence_output.overall_stance
            if evidence_output.has_relevant_evidence:
                evidence_summary = evidence_output.summary

    return SearchResult(
        title=item.title,
        url=item.url,
        snippet=item.snippet,
        engine=item.engine,
        reliability=reliability,
        evidence_summary=evidence_summary,
        evidence_overall_stance=overall_stance,
        content_characters=len(page_text),
        content_source="snippet_fallback" if not page_text else "scraped",
    )


async def search_online_async(
    claim: str,
    query: str,
    limit: int = 5,
    *,
    headless: bool = True,
    claim_index: int | None = None,
    query_index: int | None = None,
    min_credibility: str = "medium",
) -> SearchResults:
    """
    Async version: Perform online search with credibility-based filtering.

    Automatically handles low-credibility sources by fetching additional results
    from Google until we have enough high-quality sources.

    Args:
        claim: The original claim being fact-checked
        query: Search query
        limit: Desired number of high-quality results
        headless: Run Selenium in headless mode
        claim_index: Optional claim index for logging
        query_index: Optional query index for logging
        min_credibility: Minimum credibility rating ("high", "medium", "low")
    """

    search_client = GoogleSearchClient()
    reliability_checker = WebsiteReliabilityChecker()
    evidence_extractor = RelevantContentExtractor()

    # Build timer prefix for hierarchical tracking
    if claim_index is not None and query_index is not None:
        prefix = f"Step 3.{claim_index}.2.{query_index}"
    else:
        prefix = "Search"

    # Step 1: Google Search with simple adaptive fetching
    # Strategy: Fetch 1 batch, if >50% unreliable fetch 1 more, then filter
    credibility_hierarchy = {"high": 3, "medium": 2, "low": 1, "unknown": 0}
    min_credibility_score = credibility_hierarchy.get(min_credibility, 2)
    min_guaranteed_sources = max(1, limit // 2)

    batch_size = limit * 2
    all_assessed_results: List[tuple[GoogleSearchHit, int, str]] = []

    # Batch 1: Initial fetch
    with timer(f"{prefix}.1.1: Google Search API (batch 1)"):
        batch_1 = await search_client.search(query, limit=batch_size)

    if not batch_1:
        return SearchResults(query=query, results=[], total_count=0)

    # Assess credibility for batch 1
    for item in batch_1:
        reliability = await asyncio.to_thread(reliability_checker.assess, item.url)
        credibility_score = credibility_hierarchy.get(reliability.rating, 0)
        all_assessed_results.append((item, credibility_score, reliability.rating))

    # Check quality: if >50% unreliable, fetch one more batch
    reliable_count = sum(
        1 for _, score, _ in all_assessed_results if score >= min_credibility_score
    )
    unreliable_count = len(all_assessed_results) - reliable_count

    if unreliable_count > len(all_assessed_results) / 2:
        _logger.info(
            f"    ⚠ {unreliable_count}/{len(all_assessed_results)} sources are low-quality "
            f"(>{len(all_assessed_results) / 2:.0f}) → Fetching 1 additional batch"
        )

        # Batch 2: Additional fetch
        with timer(f"{prefix}.1.2: Google Search API (batch 2)"):
            batch_2 = await search_client.search(query, limit=batch_size)

        if batch_2:
            for item in batch_2:
                reliability = await asyncio.to_thread(
                    reliability_checker.assess, item.url
                )
                credibility_score = credibility_hierarchy.get(reliability.rating, 0)
                all_assessed_results.append(
                    (item, credibility_score, reliability.rating)
                )

            reliable_count = sum(
                1
                for _, score, _ in all_assessed_results
                if score >= min_credibility_score
            )
            _logger.info(
                f"    ✓ After batch 2: {reliable_count}/{len(all_assessed_results)} reliable sources"
            )
    else:
        _logger.info(
            f"    ✓ Quality acceptable: {reliable_count}/{len(all_assessed_results)} reliable "
            f"(≤50% unreliable) → Proceeding with filtering"
        )

    if not all_assessed_results:
        return SearchResults(query=query, results=[], total_count=0)

    # Step 1.5: Smart filtering with minimum guarantee
    _logger.info(
        f"    🔍 Filtering {len(all_assessed_results)} results "
        f"(min quality: {min_credibility}, min guarantee: {min_guaranteed_sources})"
    )

    # Sort by credibility score (best first)
    all_assessed_results.sort(key=lambda x: x[1], reverse=True)

    # Split into reliable and unreliable
    reliable = [r[0] for r in all_assessed_results if r[1] >= min_credibility_score]
    unreliable = [r[0] for r in all_assessed_results if r[1] < min_credibility_score]

    # Decision logic:
    if len(reliable) >= min_guaranteed_sources:
        # We have enough reliable sources - use only those
        filtered_results = reliable[: limit * 2]
        skipped = len(unreliable)
        _logger.info(
            f"    ✓ Using {len(filtered_results)} reliable sources "
            f"(filtered {skipped} low-credibility)"
        )
    else:
        # Not enough reliable sources - keep best available
        # Use ALL reliable + fill gap with best unreliable
        needed_unreliable = min_guaranteed_sources - len(reliable)
        filtered_results = reliable + unreliable[:needed_unreliable]

        _logger.warning(
            f"    ⚠ Only {len(reliable)} reliable sources found "
            f"(target: {limit}, min guarantee: {min_guaranteed_sources})"
        )
        _logger.info(
            f"    ➜ Keeping {len(filtered_results)} total sources "
            f"({len(reliable)} reliable + {len(filtered_results) - len(reliable)} best available)"
        )

    # Only process filtered sources from here
    raw_results = filtered_results

    # Initialize Selenium fetcher
    fetcher: Optional[SeleniumContentFetcher] = None
    try:
        fetcher = SeleniumContentFetcher(headless=headless)
        fetcher.__enter__()  # Initialize driver
    except RuntimeError as exc:
        _logger.warning("Selenium unavailable: %s", exc)

    try:
        # Step 2: Process all search results in parallel
        _logger.info(
            f"    ⚡ Processing {len(raw_results)} search results in PARALLEL (target: {limit} accessible)"
        )
        tasks = [
            _process_search_result_async(
                item,
                claim,
                query,
                fetcher,
                reliability_checker,
                evidence_extractor,
                prefix,
                idx,
            )
            for idx, item in enumerate(raw_results, 1)
        ]
        all_results = await asyncio.gather(*tasks)

        # Step 3: Filter unclear stances if more than half have definitive stances
        definitive_results = [
            r
            for r in all_results
            if r.evidence_overall_stance
            and r.evidence_overall_stance.lower() in ["supports", "refutes", "mixed"]
        ]
        unclear_results = [
            r
            for r in all_results
            if not r.evidence_overall_stance
            or r.evidence_overall_stance.lower() not in ["supports", "refutes", "mixed"]
        ]

        # If more than half have definitive stances, filter out unclear
        if len(definitive_results) > len(all_results) / 2:
            filtered_count = len(unclear_results)
            all_results = definitive_results
            if filtered_count > 0:
                _logger.info(
                    f"    ✓ Filtered {filtered_count} unclear stance source(s) "
                    f"({len(definitive_results)} definitive / {len(definitive_results) + filtered_count} total)"
                )
        else:
            _logger.info(
                f"    ✓ Keeping all stances: {len(definitive_results)} definitive, "
                f"{len(unclear_results)} unclear (≤50% definitive)"
            )

        # Return up to limit results
        final_results = all_results[:limit]

        return SearchResults(
            query=query, results=final_results, total_count=len(final_results)
        )
    finally:
        if fetcher:
            fetcher.__exit__(None, None, None)  # Cleanup driver
        await search_client.close()
