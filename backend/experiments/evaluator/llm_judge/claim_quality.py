from pydantic import BaseModel, Field
from experiments.evaluator.llm_judge.base import create_simple_judge


class ClaimQualityScore(BaseModel):
    """Score for a single claim's quality."""

    clarity_score: float = Field(
        ge=0.0,
        le=1.0,
        description="How clear and specific is the claim? (0.0-1.0)",
    )
    checkability_score: float = Field(
        ge=0.0,
        le=1.0,
        description="How checkable/verifiable is the claim? (0.0-1.0)",
    )
    completeness_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Can the claim be understood standalone? (0.0-1.0)",
    )
    overall_quality: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall claim quality score (0.0-1.0)",
    )
    reasoning: str = Field(description="Brief explanation of the scores")


CLAIM_QUALITY_SYSTEM_PROMPT = """You are an expert evaluator of factual claims in fact-checking systems.

Your task is to evaluate the QUALITY of extracted claims based on:

1. **Clarity** (0.0-1.0): Is the claim clear, specific, and unambiguous?
   - High: "The Earth's average temperature has increased by 1.1°C since 1880"
   - Low: "Climate change is happening"

2. **Checkability** (0.0-1.0): Can this claim be fact-checked with evidence?
   - High: Factual statements with verifiable data
   - Low: Opinions, predictions, or vague statements

3. **Completeness** (0.0-1.0): Can the claim be understood without additional context?
   - High: Standalone, includes necessary context
   - Low: Requires video context to understand

4. **Overall Quality** (0.0-1.0): Holistic assessment combining above factors

Provide scores between 0.0 (poor) and 1.0 (excellent), and a brief reasoning."""


class ClaimQualityJudge:
    """Judge for evaluating claim extraction quality."""

    def __init__(self):
        self.judge = create_simple_judge(
            response_model=ClaimQualityScore,
            system_prompt=CLAIM_QUALITY_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=300,
        )

    def evaluate(
        self, claim_text: str, transcript_excerpt: str = ""
    ) -> ClaimQualityScore:
        """
        Evaluate quality of an extracted claim.

        Args:
            claim_text: The extracted claim to evaluate
            transcript_excerpt: Optional context from transcript

        Returns:
            ClaimQualityScore with detailed scores and reasoning
        """
        prompt = f"""Evaluate the quality of this claim:

**Claim:** {claim_text}"""

        if transcript_excerpt:
            prompt += f"\n\n**Context (from transcript):** {transcript_excerpt}"

        return self.judge.evaluate_sync(prompt)
