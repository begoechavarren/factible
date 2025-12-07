import logging

from pydantic_ai import Agent

from factible.components.online_search.schemas.evidence import EvidenceExtraction
from factible.evaluation.pydantic_monitor import track_pydantic
from factible.models.config import EVIDENCE_EXTRACTOR_MODEL
from factible.models.llm import ModelChoice, ModelSpec, get_model

_logger = logging.getLogger(__name__)


class RelevantContentExtractor:
    """Use an LLM to highlight passages relevant to the query."""

    def __init__(
        self,
        model: ModelSpec | ModelChoice | None = None,
        max_characters: int = 6000,
    ) -> None:
        self.max_characters = max_characters
        if isinstance(model, ModelChoice):
            selected_model = get_model(model)
        elif model is None:
            selected_model = get_model(EVIDENCE_EXTRACTOR_MODEL)
        else:
            selected_model = model
        self._agent = Agent(
            model=selected_model,
            output_type=EvidenceExtraction,  # type: ignore[arg-type]
            system_prompt="""
            You assist a fact-checking analyst. Analyse the provided webpage content and
            identify at most three short verbatim snippets that directly relate to the
            query. For each snippet, determine its stance:

            - SUPPORTS: The snippet confirms, validates, or provides evidence for the claim
            - REFUTES: The snippet contradicts, disproves, or provides counter-evidence
            - MIXED: The snippet contains both supporting and refuting elements
            - UNCLEAR: Use ONLY when the snippet is topically relevant but genuinely ambiguous
              (e.g., discusses the topic without taking a position, uses vague language that
              could support either side, or contains contradictory statements)

            Important: If a snippet provides numerical data, facts, or statements that align
            with the claim, mark it as SUPPORTS even if the wording differs slightly from
            the original claim. Do NOT mark clear evidence as UNCLEAR.

            Provide a brief summary of what the evidence collectively indicates and set
            overall_stance accordingly.

            Use lowercase stance values: "supports", "refutes", "mixed", or "unclear".

            If no relevant evidence is found, set has_relevant_evidence to false, return
            an UNCLEAR overall_stance, and do not include snippets.
            """,
        )

    @track_pydantic("evidence_extraction")
    async def extract(self, query: str, content: str) -> EvidenceExtraction:
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
            result = await self._agent.run(prompt)
            return result.output
        except Exception as exc:
            _logger.error("Evidence extraction failed: %s", exc)
            return EvidenceExtraction(
                has_relevant_evidence=False,
                summary=None,
                snippets=[],
                overall_stance="unclear",
            )
