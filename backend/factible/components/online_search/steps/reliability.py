import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

from factible.components.online_search.schemas.reliability import SiteReliability

import whois

_logger = logging.getLogger(__name__)


class WebsiteReliabilityChecker:
    """Estimate the reliability of a domain using MBFC data and basic heuristics."""

    HIGH_TRUST_TLDS = (".gov", ".edu", ".int")

    def __init__(self, dataset_path: Optional[str] = None) -> None:
        self._dataset = self._load_dataset(dataset_path)

    def assess(self, url: str) -> SiteReliability:
        """
        Assess source reliability using MBFC credibility ratings and basic heuristics.

        Priority order:
        1. MBFC Credibility rating (if available) - authoritative
        2. Government/education TLD (.gov, .edu, .int) - high trust
        3. Domain age - older domains more established
        """
        domain = self._extract_domain(url)
        if not domain:
            return SiteReliability(
                rating="unknown",
                score=0.5,
                reasons=["Unable to parse domain"],
                bias=None,
            )

        reasons: List[str] = []

        # Check more authoritative MBFC dataset first
        dataset_entry = self._dataset.get(domain)
        if dataset_entry:
            credibility = (dataset_entry.get("credibility") or "").lower()
            factual = dataset_entry.get("factual", "")
            bias = dataset_entry.get("bias", "")

            # Map MBFC credibility directly to the rating system
            credibility_map = {
                "high": ("high", 0.85),
                "medium": ("medium", 0.60),
                "low": ("low", 0.30),
                "very low": ("low", 0.15),
            }

            if credibility in credibility_map:
                rating, score = credibility_map[credibility]
                reasons.append(f"MBFC credibility: {credibility}")
                if factual:
                    reasons.append(f"Factual reporting: {factual}")
                return SiteReliability(
                    rating=rating, score=score, reasons=reasons, bias=bias or None
                )

        # Back to heuristics if not in MBFC dataset
        score = 0.5

        # Government/education domains
        if any(domain.endswith(tld) for tld in self.HIGH_TRUST_TLDS):
            score = 0.90
            reasons.append("Government or educational institution")

        # Domain age as additional signal
        age_years = self._domain_age(domain)
        if age_years is not None:
            if age_years >= 10:
                score += 0.10
                reasons.append(f"Established domain ({age_years:.1f} years)")
            elif age_years < 1:
                score -= 0.15
                reasons.append("New domain (less than 1 year)")

        # Clamp score
        score = min(1.0, max(0.0, score))

        # Determine rating from score
        if score >= 0.75:
            rating = "high"
        elif score >= 0.50:
            rating = "medium"
        else:
            rating = "low"

        if not reasons:
            reasons.append("No credibility data available")
            rating = "unknown"

        return SiteReliability(rating=rating, score=score, reasons=reasons, bias=None)

    @staticmethod
    def _extract_domain(url: str) -> str:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain

    def _load_dataset(self, dataset_path: Optional[str]) -> dict[str, dict]:
        candidate_paths: List[Path] = []
        if dataset_path:
            candidate_paths.append(Path(dataset_path))

        media_bias_dir = Path(__file__).resolve().parent / "media_bias"
        candidate_paths.extend(self._find_dataset_snapshots(media_bias_dir))

        for path in candidate_paths:
            if path.exists():
                try:
                    with path.open("r", encoding="utf-8") as handle:
                        dataset = json.load(handle)
                    mapped: dict[str, dict] = {}
                    for entry in dataset:
                        url_or_domain = (
                            entry.get("url")
                            or entry.get("domain")
                            or entry.get("Source URL")
                        )
                        if not url_or_domain:
                            continue
                        domain = self._extract_domain(url_or_domain)
                        if domain and domain.lower() != "dead":
                            normalized_entry = {
                                "credibility": entry.get("credibility")
                                or entry.get("Credibility"),
                                "factual": entry.get("factual")
                                or entry.get("Factual Reporting"),
                                "bias": entry.get("bias") or entry.get("Bias"),
                            }
                            mapped[domain] = normalized_entry
                    if mapped:
                        _logger.info(f"Loaded media bias dataset from {path}")
                    return mapped
                except Exception as exc:
                    _logger.warning(f"Failed to load media bias dataset {path}: {exc}")
        return {}

    @staticmethod
    def _find_dataset_snapshots(directory: Path) -> List[Path]:
        snapshots = sorted(
            (
                path
                for path in directory.glob("*_media_bias_data.json")
                if path.is_file()
            ),
            reverse=True,
        )

        legacy_file = directory / "media_bias_data.json"
        if legacy_file.exists():
            snapshots.append(legacy_file)

        if not snapshots:
            snapshots.append(legacy_file)

        return snapshots

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
