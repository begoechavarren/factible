import logging

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models import ModelSettings

from factible.components.claim_extractor.schemas import Claim
from factible.components.query_generator.schemas import GeneratedQueries
from factible.tracking.pydantic_monitor import track_pydantic
from factible.models.config import QUERY_GENERATOR_MODEL
from factible.models.llm import get_model

_logger = logging.getLogger(__name__)


class QueryGeneratorDeps(BaseModel):
    max_queries: int | None = None


QUERY_GENERATOR_MODEL_SETTINGS: ModelSettings = {
    "temperature": 0.0,
    "max_tokens": 600,
}


def _get_query_generator_agent() -> Agent[QueryGeneratorDeps]:
    """Get the query generator agent instance."""
    agent = Agent(
        model=get_model(QUERY_GENERATOR_MODEL),
        output_type=GeneratedQueries,  # type: ignore[arg-type]
        deps_type=QueryGeneratorDeps,
        model_settings=QUERY_GENERATOR_MODEL_SETTINGS,
        system_prompt="""
        You are an expert fact-checker specializing in generating effective search queries.
        Your task is to create multiple search queries that will help verify or refute factual claims.
        The input will include the claim text and a short context note (speaker, timeframe, event).
        Your queries must respect the context when choosing keywords (e.g., include relevant years or
        qualifiers if provided, avoid mixing eras when context specifies a timeframe).

        For each claim, generate up to the requested number of search queries with these types:

        1. DIRECT: Search for the exact claim or very similar phrasing
        2. ALTERNATIVE: Rephrase the claim using different keywords/terms
        3. SOURCE: Look for authoritative sources that might contain this information
        4. CONTEXT: Search for broader context or related information

        Query generation principles:
        - Use specific, searchable keywords
        - Include relevant dates, names, numbers when present
        - Consider different phrasings and synonyms
        - Think about what authoritative sources might say
        - Include potential counter-arguments or opposing views

        Prioritize queries by likelihood of finding reliable information:
        - Priority 1: Most likely to find definitive information
        - Priority 2: Good chance of finding relevant information
        - Priority 3: Moderate chance, worth trying
        - Priority 4: Lower chance but might provide context
        - Priority 5: Least likely but could be useful for completeness

        Keep queries concise but specific enough to be effective. Return only the query, query_type, and
        priority fields—do not include explanations or rationale. Order your output from the highest
        priority (1) to the lowest value provided.
        """,
    )

    @agent.instructions
    def _limit_instruction(ctx: RunContext[QueryGeneratorDeps]) -> str:
        max_queries = ctx.deps.max_queries
        if max_queries is not None and max_queries >= 0:
            return f"Return no more than {max_queries} total queries for this claim."
        return "Return only the highest-priority queries needed for verification."

    return agent


@track_pydantic("query_generation")
async def generate_queries(
    claim: Claim,
    *,
    max_queries: int | None = None,
    priority_threshold: int = 2,
) -> GeneratedQueries:
    """Generate search queries for fact-checking a claim (async)."""
    agent = _get_query_generator_agent()
    context_note = (
        f"Context: {claim.context}" if claim.context else "Context: unspecified"
    )
    prompt = (
        "Generate effective search queries to fact-check this claim."
        f"\nClaim: {claim.text}\n{context_note}"
        "\nFocus on the timeframe implied by the context, if any."
        "\nReturn only the structured fields—no explanations."
    )
    result = await agent.run(
        prompt,
        deps=QueryGeneratorDeps(max_queries=max_queries),
    )
    generated = result.output

    filtered_queries = [
        query for query in generated.queries if query.priority <= priority_threshold
    ]
    filtered_queries.sort(key=lambda query: query.priority)

    if max_queries is not None and max_queries >= 0:
        filtered_queries = filtered_queries[:max_queries]

    return GeneratedQueries(
        original_claim=generated.original_claim,
        queries=filtered_queries,
        total_count=len(generated.queries),
    )
