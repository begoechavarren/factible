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
            identify at most three verbatim snippets that directly relate to the query.
            For each snippet, determine its stance:

            - SUPPORTS: The snippet confirms, validates, or provides evidence for the claim.
              This includes:
              * Direct statements that validate the claim
              * Descriptions of the same mechanism even without using exact terminology
                (e.g., "gases trap heat" = greenhouse effect, even if phrase not used)
              * Evidence of causal chains mentioned in the claim
              * Related facts that support the underlying assertion
              * Semantic equivalents (e.g., "fossil fuels cause emissions" supports
                "greenhouse effect causes warming" because fossil fuels → greenhouse gases
                → greenhouse effect)

            - REFUTES: The snippet contradicts, disproves, or provides counter-evidence that
              directly challenges the claim or its core mechanism.

            - MIXED: The snippet contains BOTH supporting AND refuting elements about the
              same claim (e.g., "X is true BUT Y contradicts it").

            - UNCLEAR: Use ONLY when the snippet is topically relevant but GENUINELY ambiguous:
              * Discusses related topics without addressing the specific claim
              * Uses vague language that could support either position
              * Lacks sufficient context to determine stance

            CRITICAL INSTRUCTIONS:
            1. Be decisive - prefer SUPPORTS or REFUTES over UNCLEAR
            2. Recognize mechanisms even without exact terminology:
               - "gases trap heat" = greenhouse effect (even if term not mentioned)
               - "CO2 absorbs radiation" = greenhouse effect
               - "emissions warm the planet" = greenhouse effect
            3. Consider causal chains: if the claim says "A causes B" and evidence says
               "C causes D" where C→A and D→B, that SUPPORTS the claim
            4. Look for semantic agreement and mechanism description, not just exact phrase matches
            5. Use the article title for context - it often clarifies ambiguous content

            Examples:
            - Claim: "Human activities cause global warming"
              Snippet: "Emissions from human activities have unequivocally caused warming"
              → SUPPORTS (direct validation)

            - Claim: "Greenhouse effect is the main cause of climate change"
              Snippet: "Fossil fuels are the largest contributor to climate change"
              → SUPPORTS (fossil fuels → greenhouse gases → greenhouse effect)

            - Claim: "Greenhouse effect causes climate change"
              Snippet: "Greenhouse gas emissions trap the sun's heat, warming the planet"
              → SUPPORTS (describes greenhouse effect mechanism even without using term)

            - Claim: "CO2 has increased 50% since pre-industrial times"
              Article Title: "CO2 now 50% higher than pre-industrial levels"
              Snippet: "The increase reached 40% by 2011, now reaching 50%"
              → SUPPORTS (title + content confirm the claim)

            Provide a brief summary of what the evidence collectively indicates and set
            overall_stance accordingly. Prefer SUPPORTS or REFUTES when evidence is clear.

            Use lowercase stance values: "supports", "refutes", "mixed", or "unclear".

            If no relevant evidence is found, set has_relevant_evidence to false, return
            an UNCLEAR overall_stance, and do not include snippets.
            """,
        )

    @track_pydantic("evidence_extraction")
    async def extract(
        self, query: str, content: str, title: str | None = None
    ) -> EvidenceExtraction:
        if not content.strip():
            return EvidenceExtraction(
                has_relevant_evidence=False,
                summary=None,
                snippets=[],
                overall_stance="unclear",
            )

        trimmed_content = content[: self.max_characters]

        # Include title if available as it often contains key context
        prompt_parts = ["Query:\n", f"{query}\n\n"]
        if title:
            prompt_parts.extend(["Article Title:\n", f"{title}\n\n"])
        prompt_parts.extend(
            ["Content:\n", f"{trimmed_content}\n\n", "Provide relevant evidence."]
        )

        prompt = "".join(prompt_parts)

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
