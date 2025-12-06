import logging
import os
from typing import List, Optional

import httpx
from pydantic import BaseModel

_logger = logging.getLogger(__name__)


class GoogleSearchHit(BaseModel):
    """Raw search hit returned by Serper."""

    title: str
    url: str
    snippet: str
    engine: str = "google-serper"


class GoogleSearchClient:
    """Wrapper around the Google Serper search API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: str = "https://google.serper.dev/search",
    ):
        self.api_key = api_key or os.getenv("SERPER_API_KEY")
        self.endpoint = endpoint
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def search(self, query: str, limit: int = 5) -> List[GoogleSearchHit]:
        if not self.api_key:
            _logger.error(
                "SERPER_API_KEY is not configured. Cannot perform Google search."
            )
            return []

        payload = {"q": query, "num": min(limit, 10)}
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }

        client = await self._get_client()

        try:
            response = await client.post(
                self.endpoint,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            _logger.error("Google search failed for '%s': %s", query, exc)
            return []

        try:
            data = response.json()
        except ValueError:
            _logger.error("Google search returned invalid JSON for '%s'", query)
            return []

        raw_results = data.get("organic") or data.get("results") or []
        hits: List[GoogleSearchHit] = []
        for entry in raw_results[:limit]:
            hits.append(
                GoogleSearchHit(
                    title=entry.get("title", ""),
                    url=entry.get("link") or entry.get("url", ""),
                    snippet=entry.get("snippet") or entry.get("description", ""),
                )
            )

        if not hits:
            _logger.info("No Google results returned for '%s'", query)

        return hits
