import asyncio
import logging
import re
from typing import Optional

from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

_logger = logging.getLogger(__name__)


class SeleniumContentFetcher:
    """Fetch textual content from web pages using Selenium."""

    def __init__(
        self,
        headless: bool = True,
        wait_timeout: int = 12,
        page_load_timeout: int = 20,
        max_characters: int = 8000,
    ) -> None:
        if not self.is_available():
            raise RuntimeError(
                "Selenium driver is not available. Install 'selenium' and ensure Chrome is present."
            )

        chrome_options = Options()  # type: ignore[call-arg]
        if headless:
            chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--blink-settings=imagesEnabled=false")

        self._options = chrome_options
        self.wait_timeout = wait_timeout
        self.page_load_timeout = page_load_timeout
        self.max_characters = max_characters
        self._driver: Optional[webdriver.Chrome] = None  # type: ignore[type-arg]
        self._wait: Optional[WebDriverWait] = None  # type: ignore[assignment]

    def __enter__(self) -> "SeleniumContentFetcher":
        self._driver = webdriver.Chrome(options=self._options)  # type: ignore[call-arg]
        self._driver.set_page_load_timeout(self.page_load_timeout)
        self._wait = WebDriverWait(self._driver, self.wait_timeout)  # type: ignore[call-arg]
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001 (dynamic typing)
        if self._driver:
            self._driver.quit()
        self._driver = None
        self._wait = None

    @staticmethod
    def is_available() -> bool:
        return (
            webdriver is not None and Options is not None and WebDriverWait is not None
        )

    def fetch_text(self, url: str, min_characters: int = 200) -> str:
        """Synchronous fetch (blocking I/O)."""
        if not self._driver:
            raise RuntimeError(
                "Selenium driver not initialised. Use SeleniumContentFetcher as a context manager."
            )

        try:
            self._driver.get(url)
            if self._wait and EC is not None:
                self._wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))  # type: ignore[arg-type]

            paragraphs = []
            if By is not None and self._driver:
                for element in self._driver.find_elements(By.TAG_NAME, "p"):
                    text = self._clean_text(element.text)
                    if text:
                        paragraphs.append(text)

            content = "\n".join(paragraphs)
            if len(content) < min_characters and self._driver and By is not None:
                try:
                    body_element = self._driver.find_element(By.TAG_NAME, "body")  # type: ignore[arg-type]
                    body_text = self._clean_text(body_element.text)
                    content = body_text
                except Exception:
                    pass

            if len(content) > self.max_characters:
                content = content[: self.max_characters]

            if len(content) < min_characters:
                _logger.debug("Content too short for %s (%d chars)", url, len(content))
                return ""

            return content
        except (TimeoutException, WebDriverException) as exc:  # type: ignore[arg-type]
            _logger.warning("Selenium failed to fetch %s: %s", url, exc)
            return ""
        except Exception as exc:
            _logger.error("Unexpected Selenium error for %s: %s", url, exc)
            return ""

    async def fetch_text_async(self, url: str, min_characters: int = 200) -> str:
        """Async wrapper for blocking Selenium fetch."""
        return await asyncio.to_thread(self.fetch_text, url, min_characters)

    @staticmethod
    def _clean_text(text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip()
