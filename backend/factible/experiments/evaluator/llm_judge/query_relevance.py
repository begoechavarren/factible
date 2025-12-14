"""
LLM-as-judge for query-claim relevance evaluation.
"""

from pydantic import BaseModel, Field
from factible.experiments.evaluator.llm_judge.base import create_simple_judge


class QueryRelevanceScore(BaseModel):
    """Score for query relevance to claim."""

    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
        description="How relevant is the query to the claim? (0.0-1.0)",
    )
    specificity_score: float = Field(
        ge=0.0,
        le=1.0,
        description="How specific/targeted is the query? (0.0-1.0)",
    )
    reasoning: str = Field(description="Brief explanation")


QUERY_RELEVANCE_SYSTEM_PROMPT = """You are an expert evaluator of search queries for fact-checking.

Evaluate how well a search query supports fact-checking a claim:

1. **Relevance** (0.0-1.0): Does the query target the right information?
   - High: Directly addresses the claim's core assertion
   - Low: Vague or off-topic

2. **Specificity** (0.0-1.0): Is the query specific enough to retrieve useful evidence?
   - High: Includes key terms, names, numbers from the claim
   - Low: Too broad or generic

Provide scores between 0.0 (poor) and 1.0 (excellent)."""


class QueryRelevanceJudge:
    """Judge for evaluating query-claim relevance."""

    def __init__(self):
        self.judge = create_simple_judge(
            response_model=QueryRelevanceScore,
            system_prompt=QUERY_RELEVANCE_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=200,
        )

    def evaluate(self, claim: str, query: str) -> QueryRelevanceScore:
        """
        Evaluate relevance of a query to its claim.

        Args:
            claim: The claim being fact-checked
            query: The search query generated

        Returns:
            QueryRelevanceScore with scores and reasoning
        """
        prompt = f"""Evaluate this search query:

**Claim:** {claim}

**Search Query:** {query}"""

        return self.judge.evaluate_sync(prompt)
