import logging

from pydantic_ai import Agent

from factible.components.online_search.schemas.evidence import EvidenceExtraction

_logger = logging.getLogger(__name__)


class RelevantContentExtractor:
    """Use an LLM to highlight passages relevant to the query."""

    def __init__(
        self, model: str = "openai:gpt-4o-mini", max_characters: int = 6000
    ) -> None:
        self.max_characters = max_characters
        self._agent = Agent(
            model=model,
            output_type=EvidenceExtraction,  # type: ignore[arg-type]
            system_prompt="""
            You assist a fact-checking analyst. Analyse the provided webpage content and
            identify at most three short verbatim snippets that directly relate to the
            query. For each snippet, record whether it SUPPORTS, REFUTES, or is MIXED
            regarding the claim. Use UNCLEAR only when relevance exists but stance cannot
            be determined. Provide a brief summary of what the evidence collectively
            indicates and set overall_stance accordingly.

            When filling the schema, use lowercase stance values: "supports", "refutes",
            "mixed", or "unclear".

            If no relevant evidence is found, set has_relevant_evidence to false, return
            an UNCLEAR overall_stance, and do not include snippets.
            """,
        )

    def extract(self, query: str, content: str) -> EvidenceExtraction:
        if not content.strip():
            return EvidenceExtraction(
                has_relevant_evidence=False,
                summary=None,
                snippets=[],
                overall_stance="unclear",
            )

        trimmed_content = content[: self.max_characters]
        prompt = (
            "Query:\n"
            f"{query}\n\n"
            "Content:\n"
            f"{trimmed_content}\n\n"
            "Provide relevant evidence."
        )

        try:
            result = self._agent.run_sync(prompt)
            return result.output
        except Exception as exc:
            _logger.error("Evidence extraction failed: %s", exc)
            return EvidenceExtraction(
                has_relevant_evidence=False,
                summary=None,
                snippets=[],
                overall_stance="unclear",
            )
