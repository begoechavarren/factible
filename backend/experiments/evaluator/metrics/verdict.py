from typing import List, Dict
import logging
import numpy as np
import asyncio

from experiments.evaluator.models import (
    VerdictAccuracyMetrics,
    GroundTruthClaim,
)
from experiments.evaluator.llm_judge.explanation_quality import ExplanationQualityJudge

_logger = logging.getLogger(__name__)

STANCE_LABELS = ["SUPPORTS", "REFUTES", "MIXED", "UNCLEAR"]


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

        # Track predictions for confusion matrix
        all_gt_stances = []
        all_sys_stances = []

        for match in matched_claims:
            gt_text = match["gt_text"]
            sys_text = match["sys_text"]

            if gt_text in gt_verdicts and sys_text in sys_verdicts:
                gt_v = gt_verdicts[gt_text]
                sys_v = sys_verdicts[sys_text]

                gt_stance = gt_v.get("overall_stance", "").upper()
                sys_stance = sys_v.get("overall_stance", "").upper()

                all_gt_stances.append(gt_stance)
                all_sys_stances.append(sys_stance)

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

        # Build confusion matrix
        confusion_matrix = self._build_confusion_matrix(all_gt_stances, all_sys_stances)

        # Calculate per-class F1 scores
        per_class_f1 = self._calculate_per_class_f1(all_gt_stances, all_sys_stances)

        # LLM-as-judge for explanation quality (optional)
        explanation_quality_avg = 0.0
        if self.enable_llm_judge and self.judge:
            _logger.info("  Running LLM-as-judge for explanation quality...")

            # Evaluate explanations in parallel (limit to avoid excessive API calls)
            max_reports = min(len(claim_reports), 5)
            quality_scores = asyncio.run(
                self._evaluate_explanations_async(claim_reports[:max_reports])
            )

            if quality_scores:
                explanation_quality_avg = np.mean(quality_scores)
                _logger.info(
                    "  Evaluated %d explanations in parallel, avg quality: %.2f",
                    len(quality_scores),
                    explanation_quality_avg,
                )

        return VerdictAccuracyMetrics(
            overall_accuracy=overall_accuracy,
            stance_accuracy=stance_accuracy,
            correct_verdicts=correct,
            total_verdicts=total,
            verdict_errors=verdict_errors,
            explanation_quality_avg=explanation_quality_avg,
            confusion_matrix=confusion_matrix,
            per_class_f1=per_class_f1,
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
                _logger.warning(f"    Failed to evaluate explanation quality: {e}")
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

    def _build_confusion_matrix(
        self, gt_stances: List[str], sys_stances: List[str]
    ) -> List[List[int]]:
        """
        Build confusion matrix from ground truth and system stances.

        Args:
            gt_stances: List of ground truth stance labels
            sys_stances: List of system-predicted stance labels

        Returns:
            Confusion matrix as list of lists [n_classes x n_classes]
            Rows = ground truth, Columns = predicted
        """
        # Initialize matrix with zeros
        n_classes = len(STANCE_LABELS)
        matrix = [[0] * n_classes for _ in range(n_classes)]

        # Create label to index mapping
        label_to_idx = {label: idx for idx, label in enumerate(STANCE_LABELS)}

        # Fill matrix
        for gt, sys in zip(gt_stances, sys_stances):
            gt_idx = label_to_idx.get(gt)
            sys_idx = label_to_idx.get(sys)
            if gt_idx is not None and sys_idx is not None:
                matrix[gt_idx][sys_idx] += 1

        return matrix

    def _calculate_per_class_f1(
        self, gt_stances: List[str], sys_stances: List[str]
    ) -> Dict[str, float]:
        """
        Calculate per-class F1 scores.

        Args:
            gt_stances: List of ground truth stance labels
            sys_stances: List of system-predicted stance labels

        Returns:
            Dictionary mapping class label to F1 score
        """
        per_class_f1 = {}

        for label in STANCE_LABELS:
            # True positives: both GT and system are this label
            tp = sum(
                1
                for gt, sys in zip(gt_stances, sys_stances)
                if gt == label and sys == label
            )
            # False positives: system predicts this label but GT is different
            fp = sum(
                1
                for gt, sys in zip(gt_stances, sys_stances)
                if gt != label and sys == label
            )
            # False negatives: GT is this label but system predicts different
            fn = sum(
                1
                for gt, sys in zip(gt_stances, sys_stances)
                if gt == label and sys != label
            )

            # Calculate precision, recall, F1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            per_class_f1[label] = f1

        return per_class_f1
