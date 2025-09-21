import logging

from pydantic_ai import Agent

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


def generate_queries(claim: str) -> GeneratedQueries:
    """Generate search queries for fact-checking a claim."""
    agent = _get_query_generator_agent()
    result = agent.run_sync(
        f"Generate effective search queries to fact-check this claim: {claim}"
    )

    return result.output
