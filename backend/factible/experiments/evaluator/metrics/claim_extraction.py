"""
Claim extraction metrics calculator.

Calculates precision, recall, F1, MAP, importance-based metrics for claim extraction.
"""

from typing import List, Dict
import numpy as np
import asyncio

from ..models import ClaimExtractionMetrics, GroundTruthClaim
from ..claim_matching import (
    semantic_similarity_match_claims,
    calculate_mean_average_precision,
)
from ..llm_judge import ClaimQualityJudge


class ClaimExtractionEvaluator:
    """Evaluates claim extraction performance."""

    def __init__(self, similarity_model=None, enable_llm_judge=False):
        """
        Initialize evaluator.

        Args:
            similarity_model: Pre-loaded sentence-transformer model (optional)
            enable_llm_judge: Whether to use LLM-as-judge for claim quality
        """
        self.similarity_model = similarity_model
        self.enable_llm_judge = enable_llm_judge
        self.judge = ClaimQualityJudge() if enable_llm_judge else None

    def evaluate(
        self,
        gt_claims: List[GroundTruthClaim],
        extracted_claims_data: List[Dict],
    ) -> ClaimExtractionMetrics:
        """
        Evaluate claim extraction with comprehensive metrics.

        Args:
            gt_claims: Ground truth claims
            extracted_claims_data: System extracted claims (list of dicts with 'text', 'importance')

        Returns:
            ClaimExtractionMetrics with all calculated metrics
        """

        # Convert dict to simple objects for matching
        class SimpleClaim:
            def __init__(self, data):
                self.text = data["text"]
                self.importance = data.get("importance", 0.0)

        system_claims = [SimpleClaim(c) for c in extracted_claims_data]

        # Match claims using semantic similarity
        matches = semantic_similarity_match_claims(
            gt_claims, system_claims, model=self.similarity_model
        )

        # Basic metrics
        tp = len(matches["true_positives"])
        fp = len(matches["false_positives"])
        fn = len(matches["false_negatives"])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # MAP (Mean Average Precision)
        map_score = calculate_mean_average_precision(system_claims, gt_claims, matches)

        # Recall@Important
        important_gt_claims = [c for c in gt_claims if c.importance >= 0.80]
        matched_important = sum(
            1
            for gt_claim, sys_claim, *_ in matches["true_positives"]
            if gt_claim.importance >= 0.80
        )
        recall_at_important = (
            matched_important / len(important_gt_claims) if important_gt_claims else 0.0
        )

        # Importance-Weighted Coverage
        matched_importance = sum(
            gt_claim.importance for gt_claim, _, *_ in matches["true_positives"]
        )
        total_importance = sum(c.importance for c in gt_claims)
        importance_weighted_coverage = (
            matched_importance / total_importance if total_importance > 0 else 0.0
        )

        # Importance MAE (Mean Absolute Error)
        importance_errors = [
            abs(gt_claim.importance - sys_claim.importance)
            for gt_claim, sys_claim, *_ in matches["true_positives"]
        ]
        importance_mae = np.mean(importance_errors) if importance_errors else 0.0

        # Detailed matching info
        matched_claims = [
            {
                "gt_text": gt_claim.claim_text,
                "sys_text": sys_claim.text,
                "gt_importance": gt_claim.importance,
                "sys_importance": sys_claim.importance,
                "importance_error": abs(gt_claim.importance - sys_claim.importance),
            }
            for gt_claim, sys_claim, *_ in matches["true_positives"]
        ]

        missed_claims = [c.claim_text for c in matches["false_negatives"]]
        extra_claims = [c.text for c in matches["false_positives"]]

        # LLM-as-judge for claim quality (optional)
        claim_quality_avg = 0.0
        if self.enable_llm_judge and self.judge:
            print("  Running LLM-as-judge for claim quality...")

            # Evaluate extracted claims in parallel (limit to avoid excessive API calls)
            max_claims_to_evaluate = min(len(system_claims), 5)
            quality_scores = asyncio.run(
                self._evaluate_claims_async(system_claims[:max_claims_to_evaluate])
            )

            if quality_scores:
                claim_quality_avg = np.mean(quality_scores)
                print(
                    f"  Evaluated {len(quality_scores)} claims in parallel, avg quality: {claim_quality_avg:.2f}"
                )

        return ClaimExtractionMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            mean_average_precision=map_score,
            recall_at_important=recall_at_important,
            importance_weighted_coverage=importance_weighted_coverage,
            matched_claims=matched_claims,
            missed_claims=missed_claims,
            extra_claims=extra_claims,
            importance_mae=importance_mae,
            claim_quality_avg=claim_quality_avg,
        )

    async def _evaluate_claims_async(self, claims: List) -> List[float]:
        """Evaluate claims in parallel using async."""

        async def evaluate_one(claim):
            try:
                score = await self.judge.judge.evaluate_async(
                    f"Evaluate the quality of this claim:\n\n**Claim:** {claim.text}"
                )
                return score.overall_quality
            except Exception as e:
                print(f"    Warning: Failed to evaluate claim quality: {e}")
                return None

        # Run all evaluations in parallel
        results = await asyncio.gather(*[evaluate_one(claim) for claim in claims])
        # Filter out None values (failed evaluations)
        return [score for score in results if score is not None]
