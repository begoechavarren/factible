import logging
import re
from typing import Optional

_logger = logging.getLogger(__name__)


class PaywallDetector:
    """Detect if fetched content is behind a paywall or restricted."""

    # Common paywall indicators in page content
    PAYWALL_INDICATORS = [
        # Generic paywall messages
        r"subscribe\s+to\s+(read|continue|access)",
        r"sign\s+in\s+to\s+(read|continue|access)",
        r"log\s+in\s+to\s+(read|continue|access)",
        r"(create|register)\s+(an\s+)?account\s+to\s+(read|continue|access)",
        r"upgrade\s+to\s+(premium|plus|pro)",
        r"become\s+a\s+(member|subscriber)",
        r"subscription\s+required",
        r"this\s+article\s+is\s+(exclusive|premium)",
        r"members[\-\s]only\s+content",
        r"you've\s+reached\s+your\s+(free\s+)?(article|story)\s+limit",
        r"unlock\s+this\s+(article|story|content)",
        # Academic/research paywalls
        r"purchase\s+(access|article|pdf)",
        r"download\s+pdf\s+\-\s+\$",
        r"access\s+through\s+your\s+institution",
        r"(abstract|summary)\s+only",
        r"full\s+text\s+available",  # Often indicates abstract-only access
        r"request\s+(full\s+)?text",
        # Newspaper paywalls
        r"free\s+trial",
        r"cancel\s+anytime",
        r"(continue|keep)\s+reading\s+with\s+(a\s+)?(free\s+)?trial",
        # ScienceDirect, Springer, etc.
        r"recommended\s+articles",  # Common on paywalled academic sites
        r"purchase\s+PDF",
        r"rent\s+this\s+article",
    ]

    # Minimum content length threshold (chars)
    # If content is very short, likely just abstract/teaser
    # Note: Academic abstracts (800-1500 chars) can still be valuable evidence
    MIN_SUBSTANTIVE_LENGTH = 300

    # Known paywalled domains (optional - can expand)
    KNOWN_PAYWALLED_DOMAINS = {
        "wsj.com",
        "ft.com",
        "economist.com",
        "nytimes.com",  # Sometimes paywalled
        "washingtonpost.com",  # Sometimes paywalled
        "sciencedirect.com",  # Academic paywall
        "springerlink.com",  # Academic paywall
        "tandfonline.com",  # Academic paywall
        "wiley.com",  # Academic paywall
    }

    # Specific patterns for academic abstracts (PubMed, etc.)
    ABSTRACT_ONLY_PATTERNS = [
        r"abstract\s*$",  # Ends with "Abstract"
        r"summary\s*$",
        r"^\s*abstract\s*:",  # Starts with "Abstract:"
        r"pmid:\s*\d+\s*$",  # PubMed ID at end
        r"doi:\s*10\.\d+",  # DOI present but no full text
    ]

    def __init__(
        self,
        min_length: int = MIN_SUBSTANTIVE_LENGTH,
        case_sensitive: bool = False,
    ):
        """
        Initialize paywall detector.

        Args:
            min_length: Minimum content length to consider substantive
            case_sensitive: Whether pattern matching is case-sensitive
        """
        self.min_length = min_length
        self.flags = 0 if case_sensitive else re.IGNORECASE

        # Compile regex patterns for performance
        self._compiled_indicators = [
            re.compile(pattern, self.flags) for pattern in self.PAYWALL_INDICATORS
        ]
        self._compiled_abstract_patterns = [
            re.compile(pattern, self.flags) for pattern in self.ABSTRACT_ONLY_PATTERNS
        ]

    def is_paywalled(
        self, content: str, url: str, title: Optional[str] = None
    ) -> tuple[bool, str]:
        """
        Detect if content is behind a paywall or restricted.

        Args:
            content: Fetched page content (plain text)
            url: URL of the page
            title: Optional page title for additional context

        Returns:
            (is_paywalled, reason) tuple
        """
        if not content or not content.strip():
            return True, "Empty content - likely blocked or failed to load"

        content_lower = content.lower()

        # Check 1: Content length (very short = likely abstract/teaser)
        if len(content.strip()) < self.min_length:
            # Extra check: if it's PubMed/academic and very short, it's abstract-only
            if any(
                domain in url.lower()
                for domain in [
                    "pubmed",
                    "ncbi.nlm.nih.gov",
                    "sciencedirect",
                    "springer",
                ]
            ):
                return (
                    True,
                    f"Content too short ({len(content)} chars) - likely abstract-only",
                )
            return (
                True,
                f"Content too short ({len(content)} chars) - likely paywall/restricted",
            )

        # NOTE: We intentionally DO NOT skip PubMed abstracts
        # Even though they're not full-text, abstracts often contain valuable
        # scientific evidence that LLMs can use for fact-checking
        # (e.g., "EMF exposure causes DNA damage" can be confirmed/refuted from abstract)

        # Check 2: Domain-based detection
        for domain in self.KNOWN_PAYWALLED_DOMAINS:
            if domain in url.lower():
                # If known paywalled domain, check if we got substantive content
                # (some articles may be free even on paywalled sites)
                if len(content.strip()) < self.min_length * 2:
                    return (
                        True,
                        f"Known paywalled domain ({domain}) with limited content",
                    )
                _logger.debug(
                    "Known paywalled domain %s, but content length suggests access (%d chars)",
                    domain,
                    len(content),
                )

        # Check 3: Pattern matching for paywall indicators
        for pattern in self._compiled_indicators:
            if pattern.search(content):
                match_text = pattern.pattern[:50]  # First 50 chars of pattern
                return True, f"Paywall indicator detected: {match_text}"

        # Check 4: Abstract-only detection for academic papers
        for pattern in self._compiled_abstract_patterns:
            if pattern.search(content):
                match_text = pattern.pattern[:50]
                return True, f"Abstract-only pattern detected: {match_text}"

        # Check 5: Heuristic - if content has "abstract" but very little else
        if "abstract" in content_lower and len(content.strip()) < self.min_length * 1.5:
            if content_lower.count("abstract") > 2:  # Multiple "abstract" mentions
                return (
                    True,
                    "Multiple 'abstract' mentions with short content - likely abstract-only",
                )

        # Passed all checks - appears to be accessible
        return False, "Content appears accessible"

    def should_skip_source(
        self, content: str, url: str, title: Optional[str] = None
    ) -> bool:
        """
        Simplified check: should we skip this source due to paywall/restrictions?

        Args:
            content: Fetched content
            url: Source URL
            title: Optional title

        Returns:
            True if source should be skipped, False otherwise
        """
        is_paywalled, reason = self.is_paywalled(content, url, title)
        if is_paywalled:
            _logger.info("Skipping paywalled/restricted source: %s - %s", url, reason)
        return is_paywalled
