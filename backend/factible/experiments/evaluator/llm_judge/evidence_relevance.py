"""
LLM-as-judge for evidence relevance evaluation.
"""

from pydantic import BaseModel, Field
from factible.experiments.evaluator.llm_judge.base import create_simple_judge


class EvidenceRelevanceScore(BaseModel):
    """Score for evidence relevance to claim."""

    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
        description="How relevant is the evidence to the claim? (0.0-1.0)",
    )
    credibility_score: float = Field(
        ge=0.0,
        le=1.0,
        description="How credible/trustworthy is the evidence source? (0.0-1.0)",
    )
    reasoning: str = Field(description="Brief explanation")


EVIDENCE_RELEVANCE_SYSTEM_PROMPT = """You are an expert evaluator of evidence in fact-checking.

Evaluate evidence retrieved for fact-checking a claim:

1. **Relevance** (0.0-1.0): Does the evidence address the claim?
   - High: Directly confirms/refutes the claim's core assertion
   - Low: Tangentially related or off-topic

2. **Credibility** (0.0-1.0): Is the source trustworthy?
   - High: Authoritative sources (scientific journals, official reports, expert consensus)
   - Medium: Reputable news, verified organizations
   - Low: Blogs, social media, anonymous sources

Provide scores between 0.0 (poor) and 1.0 (excellent)."""


class EvidenceRelevanceJudge:
    """Judge for evaluating evidence quality."""

    def __init__(self):
        self.judge = create_simple_judge(
            response_model=EvidenceRelevanceScore,
            system_prompt=EVIDENCE_RELEVANCE_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=200,
        )

    def evaluate(
        self, claim: str, evidence_summary: str, source_url: str = ""
    ) -> EvidenceRelevanceScore:
        """
        Evaluate relevance and credibility of evidence.

        Args:
            claim: The claim being fact-checked
            evidence_summary: Summary of the evidence found
            source_url: URL of the evidence source (optional)

        Returns:
            EvidenceRelevanceScore with scores and reasoning
        """
        prompt = f"""Evaluate this evidence:

**Claim:** {claim}

**Evidence Summary:** {evidence_summary}"""

        if source_url:
            prompt += f"\n\n**Source:** {source_url}"

        return self.judge.evaluate_sync(prompt)
