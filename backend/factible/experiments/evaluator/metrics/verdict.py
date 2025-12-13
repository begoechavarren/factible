"""
Verdict accuracy metrics calculator.
"""

from typing import List, Dict
import numpy as np
import asyncio

from ..models import VerdictAccuracyMetrics, GroundTruthClaim
from ..llm_judge import ExplanationQualityJudge


class VerdictEvaluator:
    """Evaluates verdict accuracy."""

    def __init__(self, enable_llm_judge: bool = False):
        """
        Initialize evaluator.

        Args:
            enable_llm_judge: Whether to use LLM-as-judge for explanation quality
        """
        self.enable_llm_judge = enable_llm_judge
        self.judge = ExplanationQualityJudge() if enable_llm_judge else None

    def evaluate(
        self,
        gt_claims: List[GroundTruthClaim],
        claim_reports: List[Dict],
        matched_claims: List[Dict],
    ) -> VerdictAccuracyMetrics:
        """
        Evaluate verdict accuracy.

        Args:
            gt_claims: Ground truth claims
            claim_reports: System verdict reports
            matched_claims: Matched claims from claim extraction evaluation

        Returns:
            VerdictAccuracyMetrics
        """
        # Build lookup for ground truth verdicts
        gt_verdicts = {c.claim_text: c.verdict for c in gt_claims}

        # Build lookup for system verdicts
        sys_verdicts = {r["claim_text"]: r for r in claim_reports}

        # Evaluate matched claims only
        correct = 0
        correct_stance = 0
        total = 0
        verdict_errors = []

        for match in matched_claims:
            gt_text = match["gt_text"]
            sys_text = match["sys_text"]

            if gt_text in gt_verdicts and sys_text in sys_verdicts:
                gt_v = gt_verdicts[gt_text]
                sys_v = sys_verdicts[sys_text]

                gt_stance = gt_v.get("overall_stance", "").upper()
                sys_stance = sys_v.get("overall_stance", "").upper()

                if gt_stance == sys_stance:
                    correct_stance += 1
                    correct += 1
                else:
                    verdict_errors.append(
                        {
                            "claim": gt_text,
                            "gt_verdict": gt_stance,
                            "sys_verdict": sys_stance,
                        }
                    )

                total += 1

        overall_accuracy = correct / total if total > 0 else 0.0
        stance_accuracy = correct_stance / total if total > 0 else 0.0

        # LLM-as-judge for explanation quality (optional)
        explanation_quality_avg = 0.0
        if self.enable_llm_judge and self.judge:
            print("  Running LLM-as-judge for explanation quality...")

            # Evaluate explanations in parallel (limit to avoid excessive API calls)
            max_reports = min(len(claim_reports), 5)
            quality_scores = asyncio.run(
                self._evaluate_explanations_async(claim_reports[:max_reports])
            )

            if quality_scores:
                explanation_quality_avg = np.mean(quality_scores)
                print(
                    f"  Evaluated {len(quality_scores)} explanations in parallel, avg quality: {explanation_quality_avg:.2f}"
                )

        return VerdictAccuracyMetrics(
            overall_accuracy=overall_accuracy,
            stance_accuracy=stance_accuracy,
            correct_verdicts=correct,
            total_verdicts=total,
            verdict_errors=verdict_errors,
            explanation_quality_avg=explanation_quality_avg,
        )

    async def _evaluate_explanations_async(
        self, claim_reports: List[Dict]
    ) -> List[float]:
        """Evaluate explanations in parallel using async."""

        async def evaluate_one_explanation(claim_text, explanation, stance):
            try:
                score = await self.judge.judge.evaluate_async(
                    f"""Evaluate this fact-check explanation:

**Claim:** {claim_text}
**Verdict:** {stance}
**Explanation:** {explanation}"""
                )
                return score.overall_quality
            except Exception as e:
                print(f"    Warning: Failed to evaluate explanation quality: {e}")
                return None

        # Collect all explanation evaluation tasks
        tasks = []
        for report in claim_reports:
            claim_text = report.get("claim_text", "")
            explanation = report.get("verdict_summary", "")
            stance = report.get("overall_stance", "")

            if claim_text and explanation:
                tasks.append(evaluate_one_explanation(claim_text, explanation, stance))

        # Evaluate all explanations in parallel
        results = await asyncio.gather(*tasks)
        # Filter out None values (failed evaluations)
        return [score for score in results if score is not None]
