from pydantic import BaseModel, Field
from experiments.evaluator.llm_judge.base import create_simple_judge


class ExplanationQualityScore(BaseModel):
    """Score for verdict explanation quality."""

    clarity_score: float = Field(
        ge=0.0,
        le=1.0,
        description="How clear and understandable is the explanation? (0.0-1.0)",
    )
    evidence_support_score: float = Field(
        ge=0.0,
        le=1.0,
        description="How well does the explanation cite and use evidence? (0.0-1.0)",
    )
    completeness_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Does the explanation address all aspects of the claim? (0.0-1.0)",
    )
    overall_quality: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall explanation quality score (0.0-1.0)",
    )
    reasoning: str = Field(description="Brief explanation of the scores")


EXPLANATION_QUALITY_SYSTEM_PROMPT = """You are an expert evaluator of fact-check explanations.

Evaluate the quality of a verdict explanation based on:

1. **Clarity** (0.0-1.0): Is the explanation clear and easy to understand?
   - High: Well-structured, uses plain language, logical flow
   - Low: Confusing, jargon-heavy, hard to follow

2. **Evidence Support** (0.0-1.0): How well does it cite and use evidence?
   - High: Cites specific sources, quotes relevant passages, explains how evidence supports conclusion
   - Low: Vague references, no citations, unsupported assertions

3. **Completeness** (0.0-1.0): Does it address all aspects of the claim?
   - High: Covers all key points, acknowledges limitations, provides context
   - Low: Misses important aspects, oversimplifies, ignores nuance

4. **Overall Quality** (0.0-1.0): Holistic assessment of explanation quality

Provide scores between 0.0 (poor) and 1.0 (excellent), and brief reasoning."""


class ExplanationQualityJudge:
    """Judge for evaluating verdict explanation quality."""

    def __init__(self):
        self.judge = create_simple_judge(
            response_model=ExplanationQualityScore,
            system_prompt=EXPLANATION_QUALITY_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=300,
        )

    def evaluate(
        self, claim: str, explanation: str, stance: str = ""
    ) -> ExplanationQualityScore:
        """
        Evaluate quality of a verdict explanation.

        Args:
            claim: The claim being fact-checked
            explanation: The verdict explanation/summary
            stance: The verdict stance (supports/refutes/unclear)

        Returns:
            ExplanationQualityScore with detailed scores and reasoning
        """
        prompt = f"""Evaluate this fact-check explanation:

**Claim:** {claim}

**Verdict:** {stance}

**Explanation:** {explanation}"""

        return self.judge.evaluate_sync(prompt)
