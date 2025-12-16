from typing import Dict

from experiments.evaluator.models import EndToEndMetrics


class EndToEndEvaluator:
    """Evaluates end-to-end system performance."""

    def evaluate(
        self,
        claim_coverage: float,
        verdict_accuracy: float,
        metrics_data: Dict,
        outputs_data: Dict,
    ) -> EndToEndMetrics:
        """
        Calculate end-to-end metrics.

        Args:
            claim_coverage: Recall from claim extraction
            verdict_accuracy: Overall verdict accuracy
            metrics_data: Metrics JSON (latency, cost)
            outputs_data: System outputs

        Returns:
            EndToEndMetrics
        """
        # Extract latency and cost from metrics
        total_time = metrics_data.get("timing", {}).get("total_seconds", 0.0)
        llm_data = metrics_data.get("llm", {})
        total_cost = llm_data.get("total_cost_usd", 0.0)

        # Calculate average evidence quality
        claim_reports = outputs_data.get("final_output", {}).get("claim_reports", [])
        evidence_quality_scores = []
        for report in claim_reports:
            if report.get("total_sources", 0) > 0:
                evidence_quality_scores.append(1.0)
            else:
                evidence_quality_scores.append(0.0)

        evidence_quality_avg = (
            sum(evidence_quality_scores) / len(evidence_quality_scores)
            if evidence_quality_scores
            else 0.0
        )

        return EndToEndMetrics(
            claim_coverage=claim_coverage,
            verdict_accuracy=verdict_accuracy,
            evidence_quality_avg=evidence_quality_avg,
            total_cost_usd=total_cost,
            total_time_seconds=total_time,
        )
