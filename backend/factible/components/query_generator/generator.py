import logging

from pydantic_ai import Agent

from factible.components.claim_extractor.schemas import Claim
from factible.components.query_generator.schemas import GeneratedQueries

_logger = logging.getLogger(__name__)


def _get_query_generator_agent() -> Agent:
    """Get the query generator agent instance."""
    return Agent(
        model="openai:gpt-4o-mini",
        output_type=GeneratedQueries,  # type: ignore[arg-type]
        system_prompt="""
        You are an expert fact-checker specializing in generating effective search queries.
        Your task is to create multiple search queries that will help verify or refute factual claims.
        The input will include the claim text and a short context note (speaker, timeframe, event).
        Your queries must respect the context when choosing keywords (e.g., include relevant years or
        qualifiers if provided, avoid mixing eras when context specifies a timeframe).

        For each claim, generate 3-5 different search queries with these types:

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

        Keep queries concise but specific enough to be effective.
        """,
    )


def generate_queries(
    claim: Claim,
    *,
    max_queries: int | None = None,
    priority_threshold: int = 2,
) -> GeneratedQueries:
    """Generate search queries for fact-checking a claim."""
    agent = _get_query_generator_agent()
    context_note = (
        f"Context: {claim.context}" if claim.context else "Context: unspecified"
    )
    prompt = (
        "Generate effective search queries to fact-check this claim:"
        f"\nClaim: {claim.text}\n{context_note}"
        "\nFocus on the timeframe implied by the context, if any."
    )
    result = agent.run_sync(prompt)
    generated = result.output

    filtered_queries = [
        query for query in generated.queries if query.priority <= priority_threshold
    ]

    if max_queries is not None and max_queries >= 0:
        filtered_queries = filtered_queries[:max_queries]

    return GeneratedQueries(
        original_claim=generated.original_claim,
        queries=filtered_queries,
        total_count=len(generated.queries),
    )
