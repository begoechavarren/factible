import asyncio
import logging
from contextlib import nullcontext
from typing import List, Optional

from factible.components.online_search.schemas.evidence import EvidenceSnippet
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
    evidence_summary = None
    overall_stance = None
    snippets: List[EvidenceSnippet] = []
    if page_text:
        with timer(f"{prefix}.4: Evidence extraction (result {result_idx})"):
            evidence_output = await evidence_extractor.extract(query, page_text)
            overall_stance = evidence_output.overall_stance
            if evidence_output.has_relevant_evidence:
                evidence_summary = evidence_output.summary
                snippets = evidence_output.snippets

    return SearchResult(
        title=item.title,
        url=item.url,
        snippet=item.snippet,
        engine=item.engine,
        reliability=reliability,
        relevant_evidence=snippets,
        evidence_summary=evidence_summary,
        evidence_overall_stance=overall_stance,
        content_characters=len(page_text),
    )


async def search_online_async(
    query: str,
    limit: int = 5,
    *,
    headless: bool = True,
    claim_index: int | None = None,
    query_index: int | None = None,
) -> SearchResults:
    """Async version: Perform online search with parallel result processing."""

    search_client = GoogleSearchClient()
    reliability_checker = WebsiteReliabilityChecker()
    evidence_extractor = RelevantContentExtractor()

    # Build timer prefix for hierarchical tracking
    if claim_index is not None and query_index is not None:
        prefix = f"Step 3.{claim_index}.2.{query_index}"
    else:
        prefix = "Search"

    # Step 1: Google Search API (async)
    with timer(f"{prefix}.1: Google Search API"):
        raw_results = await search_client.search(query, limit)

    if not raw_results:
        return SearchResults(query=query, results=[], total_count=0)

    # Initialize Selenium fetcher
    fetcher: Optional[SeleniumContentFetcher] = None
    try:
        fetcher = SeleniumContentFetcher(headless=headless)
        fetcher.__enter__()  # Initialize driver
    except RuntimeError as exc:
        _logger.warning("Selenium unavailable: %s", exc)

    try:
        # Step 2: Process all search results in parallel
        _logger.info(f"    ⚡ Processing {len(raw_results)} search results in PARALLEL")
        tasks = [
            _process_search_result_async(
                item,
                query,
                fetcher,
                reliability_checker,
                evidence_extractor,
                prefix,
                idx,
            )
            for idx, item in enumerate(raw_results, 1)
        ]
        results = await asyncio.gather(*tasks)

        return SearchResults(
            query=query, results=list(results), total_count=len(results)
        )
    finally:
        if fetcher:
            fetcher.__exit__(None, None, None)  # Cleanup driver
        await search_client.close()


def search_online(
    query: str,
    limit: int = 5,
    *,
    headless: bool = True,
    claim_index: int | None = None,
    query_index: int | None = None,
) -> SearchResults:
    """Sync version (kept for backward compatibility)."""

    search_client = GoogleSearchClient()
    reliability_checker = WebsiteReliabilityChecker()
    evidence_extractor = RelevantContentExtractor()

    # Build timer prefix for hierarchical tracking
    if claim_index is not None and query_index is not None:
        prefix = f"Step 3.{claim_index}.2.{query_index}"
    else:
        prefix = "Search"

    with timer(f"{prefix}.1: Google Search API"):
        raw_results = asyncio.run(search_client.search(query, limit))
    if not raw_results:
        return SearchResults(query=query, results=[], total_count=0)

    fetcher_cm: Optional[SeleniumContentFetcher]
    try:
        fetcher_cm = SeleniumContentFetcher(headless=headless)
    except RuntimeError as exc:
        _logger.warning("Selenium unavailable: %s", exc)
        fetcher_cm = None

    context_manager = fetcher_cm if fetcher_cm is not None else nullcontext()
    results: List[SearchResult] = []

    with context_manager as active_fetcher:  # type: ignore[assignment]
        for result_idx, item in enumerate(raw_results, 1):
            with timer(f"{prefix}.2: Reliability assessment (result {result_idx})"):
                reliability = reliability_checker.assess(item.url)

            page_text = ""
            if isinstance(active_fetcher, SeleniumContentFetcher):
                with timer(f"{prefix}.3: Content fetching (result {result_idx})"):
                    page_text = active_fetcher.fetch_text(item.url)

            evidence_summary = None
            overall_stance = None
            snippets: List[EvidenceSnippet] = []
            if page_text:
                with timer(f"{prefix}.4: Evidence extraction (result {result_idx})"):
                    evidence_output = asyncio.run(
                        evidence_extractor.extract(query, page_text)
                    )
                    overall_stance = evidence_output.overall_stance
                    if evidence_output.has_relevant_evidence:
                        evidence_summary = evidence_output.summary
                        snippets = evidence_output.snippets

            results.append(
                SearchResult(
                    title=item.title,
                    url=item.url,
                    snippet=item.snippet,
                    engine=item.engine,
                    reliability=reliability,
                    relevant_evidence=snippets,
                    evidence_summary=evidence_summary,
                    evidence_overall_stance=overall_stance,
                    content_characters=len(page_text),
                )
            )

    return SearchResults(query=query, results=results, total_count=len(results))
