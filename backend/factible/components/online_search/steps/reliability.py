import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

from factible.components.online_search.schemas.reliability import SiteReliability

import whois  # type: ignore

_logger = logging.getLogger(__name__)


class WebsiteReliabilityChecker:
    """Estimate the reliability of a domain using lightweight heuristics."""

    HIGH_TRUST_TLDS = (".gov", ".edu", ".int")
    SUPPORTIVE_TLDS = (".org", ".mil")
    LOWER_TRUST_TLDS = (".info", ".biz", ".click")

    TRUSTED_DOMAINS = (
        "who.int",
        "nih.gov",
        "cdc.gov",
        "un.org",
        "worldbank.org",
        "nature.com",
        "science.org",
        "reuters.com",
        "apnews.com",
    )

    SUSPECT_DOMAINS = (
        "wikipedia.org",
        "blogspot.com",
        "medium.com",
    )

    def __init__(self, dataset_path: Optional[str] = None) -> None:
        self._dataset = self._load_dataset(dataset_path)

    def assess(self, url: str) -> SiteReliability:
        domain = self._extract_domain(url)
        if not domain:
            return SiteReliability(
                rating="unknown", score=0.5, reasons=["Unable to parse domain"]
            )

        score = 0.5
        reasons: List[str] = []

        dataset_entry = self._dataset.get(domain)
        if dataset_entry:
            factual = (dataset_entry.get("factual") or "").lower()
            bias = (dataset_entry.get("bias") or "").lower()
            if factual in {"very high", "high", "mostly factual"}:
                score += 0.25
                reasons.append(f"Dataset factuality: {factual}")
            elif factual in {"mixed", "low", "very low"}:
                score -= 0.25
                reasons.append(f"Dataset factuality: {factual}")

            if bias in {"least biased", "left-center", "right-center", "pro-science"}:
                score += 0.1
                reasons.append(f"Dataset bias: {bias}")
            elif bias in {"extreme", "conspiracy", "questionable"}:
                score -= 0.15
                reasons.append(f"Dataset bias: {bias}")

        if any(domain.endswith(tld) for tld in self.HIGH_TRUST_TLDS):
            score += 0.2
            reasons.append("High-trust top-level domain")
        elif any(domain.endswith(tld) for tld in self.SUPPORTIVE_TLDS):
            score += 0.1
            reasons.append("Non-profit or military TLD")
        elif any(domain.endswith(tld) for tld in self.LOWER_TRUST_TLDS):
            score -= 0.1
            reasons.append("Lower-trust top-level domain")

        if any(domain.endswith(trusted) for trusted in self.TRUSTED_DOMAINS):
            score += 0.2
            reasons.append("Domain recognised as high-trust")

        if any(domain.endswith(suspect) for suspect in self.SUSPECT_DOMAINS):
            score -= 0.1
            reasons.append("Domain known for mixed reliability")

        age_years = self._domain_age(domain)
        if age_years is not None:
            if age_years >= 10:
                score += 0.15
                reasons.append(f"Established domain ({age_years:.1f} years)")
            elif age_years >= 5:
                score += 0.1
                reasons.append(f"Mature domain ({age_years:.1f} years)")
            elif age_years < 1:
                score -= 0.2
                reasons.append("Domain is less than a year old")

        score = min(1.0, max(0.0, score))

        if not reasons:
            return SiteReliability(rating="unknown", score=score, reasons=[])

        if score >= 0.75:
            rating = "high"
        elif score >= 0.55:
            rating = "medium"
        elif score >= 0.35:
            rating = "low"
        else:
            rating = "unknown"

        return SiteReliability(rating=rating, score=score, reasons=reasons)

    @staticmethod
    def _extract_domain(url: str) -> str:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain

    def _load_dataset(self, dataset_path: Optional[str]) -> dict[str, dict]:
        candidate_paths: List[Path] = []
        if dataset_path:
            candidate_paths.append(Path(dataset_path))
        default_path = (
            Path(__file__).resolve().parent.parent / "data" / "media_bias_data.json"
        )
        candidate_paths.append(default_path)

        for path in candidate_paths:
            if path.exists():
                try:
                    with path.open("r", encoding="utf-8") as handle:
                        dataset = json.load(handle)
                    mapped: dict[str, dict] = {}
                    for entry in dataset:
                        url_or_domain = entry.get("url") or entry.get("domain")
                        if not url_or_domain:
                            continue
                        domain = self._extract_domain(url_or_domain)
                        mapped[domain] = entry
                    if mapped:
                        _logger.info("Loaded media bias dataset from %s", path)
                    return mapped
                except Exception as exc:
                    _logger.warning(
                        "Failed to load media bias dataset %s: %s", path, exc
                    )
        return {}

    @staticmethod
    def _domain_age(domain: str) -> Optional[float]:
        if whois is None:
            return None
        try:
            info = whois.whois(domain)
        except Exception:
            return None

        creation_date = getattr(info, "creation_date", None)
        if isinstance(creation_date, list):
            creation_date = creation_date[0]

        if not isinstance(creation_date, datetime):
            return None

        delta = datetime.utcnow() - creation_date
        return delta.days / 365.25
