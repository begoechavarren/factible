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
            You assist a fact-checking analyst. You will receive:
            - The CLAIM to fact-check
            - Article title (if available)
            - Google Search snippet (short preview from search results)
            - Full page content (scraped from the webpage)

            Your task: Analyze ALL provided information and determine:
            1. Does this source contain relevant evidence about the claim?
            2. What is the overall stance of the evidence towards the claim?
            3. A brief synthesis explaining the relationship

            STANCE DEFINITIONS (relative to the CLAIM, not any query):

            - SUPPORTS: The evidence confirms, validates, or provides support for the claim.
              This includes:
              * Direct statements that validate the claim
              * Descriptions of the same mechanism even without exact terminology
                (e.g., "gases trap heat" = greenhouse effect)
              * Evidence of causal chains mentioned in the claim
              * Semantic equivalents that support the underlying assertion

            - REFUTES: The evidence contradicts, disproves, or challenges the claim.
              Pay special attention to qualifiers like "only", "never", "always" in claims.

            - MIXED: The evidence contains BOTH supporting AND refuting elements.

            - UNCLEAR: Use ONLY when genuinely ambiguous:
              * Discusses related topics without addressing the specific claim
              * Lacks sufficient context to determine stance
              * Content is too low-quality (navigation menus, forms, etc.)

            CRITICAL INSTRUCTIONS:
            1. Be decisive - prefer SUPPORTS or REFUTES over UNCLEAR
            2. Recognize mechanisms even without exact terminology
            3. Consider causal chains and semantic equivalents
            4. Pay attention to claim qualifiers ("only", "since X date", etc.)
            5. Use both the Google snippet AND page content - prioritize whichever has better evidence
            6. If page content is just navigation/forms/menus, rely on the Google snippet

            OUTPUT:
            - has_relevant_evidence: true if you found usable evidence, false otherwise
            - summary: 1-2 sentences explaining what the evidence says about the claim
            - overall_stance: "supports", "refutes", "mixed", or "unclear"
            - key_quote: (optional) One compelling verbatim quote if available

            If no relevant evidence is found, set has_relevant_evidence to false and overall_stance to "unclear".
            """,
        )

    @track_pydantic("evidence_extraction")
    async def extract(
        self,
        claim: str,
        query: str,
        content: str,
        title: str | None = None,
        snippet: str | None = None,
    ) -> EvidenceExtraction:
        if not content.strip():
            return EvidenceExtraction(
                has_relevant_evidence=False,
                summary=None,
                snippets=[],
                overall_stance="unclear",
            )

        trimmed_content = content[: self.max_characters]

        # Include claim, title, and snippet for proper context
        prompt_parts = ["Claim:\n", f"{claim}\n\n"]
        if title:
            prompt_parts.extend(["Article Title:\n", f"{title}\n\n"])
        if snippet:
            prompt_parts.extend(["Google Search Snippet:\n", f"{snippet}\n\n"])
        prompt_parts.extend(
            [
                "Full Page Content:\n",
                f"{trimmed_content}\n\n",
                "Provide relevant evidence.",
            ]
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
