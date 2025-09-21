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
from factible.components.online_search.steps.google_search import GoogleSearchClient
from factible.components.online_search.steps.reliability import (
    WebsiteReliabilityChecker,
)

_logger = logging.getLogger(__name__)


def search_online(
    query: str, limit: int = 5, *, headless: bool = True
) -> SearchResults:
    """Perform an online search and enrich results with evidence and reliability."""

    search_client = GoogleSearchClient()
    reliability_checker = WebsiteReliabilityChecker()
    evidence_extractor = RelevantContentExtractor()

    raw_results = search_client.search(query, limit)
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
        for item in raw_results:
            reliability = reliability_checker.assess(item.url)
            page_text = ""
            if isinstance(active_fetcher, SeleniumContentFetcher):
                page_text = active_fetcher.fetch_text(item.url)

            evidence_summary = None
            overall_stance = None
            snippets: List[EvidenceSnippet] = []
            if page_text:
                evidence_output = evidence_extractor.extract(query, page_text)
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
