from typing import Dict, List
import logging
import numpy as np
import asyncio

from experiments.evaluator.models import EvidenceSearchMetrics
from experiments.evaluator.llm_judge.evidence_relevance import EvidenceRelevanceJudge

_logger = logging.getLogger(__name__)


class EvidenceSearchEvaluator:
    """Evaluates evidence search quality."""

    def __init__(self, enable_llm_judge: bool = False):
        """
        Initialize evaluator.

        Args:
            enable_llm_judge: Whether to use LLM-as-judge
        """
        self.enable_llm_judge = enable_llm_judge
        self.judge = EvidenceRelevanceJudge() if enable_llm_judge else None

    def evaluate(self, outputs_data: Dict) -> EvidenceSearchMetrics:
        """
        Evaluate evidence search.

        Args:
            outputs_data: System outputs

        Returns:
            EvidenceSearchMetrics
        """
        claim_reports = outputs_data.get("final_output", {}).get("claim_reports", [])

        # Source reliability distribution
        reliability_dist: dict = {}
        total_evidence = 0

        for report in claim_reports:
            total_evidence += report.get("total_sources", 0)

            # Track reliability distribution
            evidence_by_stance = report.get("evidence_by_stance", {})
            for stance, evidence_list in evidence_by_stance.items():
                for evidence in evidence_list:
                    reliability = evidence.get("reliability", {})
                    rating = reliability.get("rating", "unknown")
                    reliability_dist[rating] = reliability_dist.get(rating, 0) + 1

        avg_evidence = (
            np.mean([r.get("total_sources", 0) for r in claim_reports])
            if claim_reports
            else 0.0
        )

        # LLM-as-judge for evidence relevance (optional)
        evidence_relevance_avg = 0.0
        if self.enable_llm_judge and self.judge:
            _logger.info("  Running LLM-as-judge for evidence relevance...")

            # Evaluate evidence in parallel (limit to avoid excessive API calls)
            max_claims = min(len(claim_reports), 5)
            relevance_scores = asyncio.run(
                self._evaluate_evidence_async(
                    claim_reports[:max_claims], max_evidence_per_claim=2
                )
            )

            if relevance_scores:
                evidence_relevance_avg = np.mean(relevance_scores)
                _logger.info(
                    "  Evaluated %d evidence pieces in parallel, avg relevance: %.2f",
                    len(relevance_scores),
                    evidence_relevance_avg,
                )

        return EvidenceSearchMetrics(
            evidence_relevance_avg=evidence_relevance_avg,
            source_reliability_distribution=reliability_dist,
            avg_evidence_per_claim=avg_evidence,
        )

    async def _evaluate_evidence_async(
        self, claim_reports: List, max_evidence_per_claim: int = 2
    ) -> List[float]:
        """Evaluate evidence in parallel using async."""

        async def evaluate_one_evidence(claim_text, evidence_snippet, source_title):
            try:
                score = await self.judge.judge.evaluate_async(
                    f"""Evaluate this evidence:

**Claim:** {claim_text}

**Evidence Summary:** {evidence_snippet}

**Source:** {source_title}"""
                )
                return score.relevance_score
            except Exception as e:
                _logger.warning(f"    Failed to evaluate evidence: {e}")
                return None

        # Collect all evidence evaluation tasks
        tasks = []
        for report in claim_reports:
            claim_text = report.get("claim_text", "")
            evidence_by_stance = report.get("evidence_by_stance", {})

            evidence_count = 0
            for stance, evidence_list in evidence_by_stance.items():
                for evidence in evidence_list:
                    if evidence_count >= max_evidence_per_claim:
                        break

                    evidence_snippet = evidence.get("snippet", "")
                    source_title = evidence.get("title", "")

                    if evidence_snippet and claim_text:
                        tasks.append(
                            evaluate_one_evidence(
                                claim_text, evidence_snippet, source_title
                            )
                        )
                        evidence_count += 1

                if evidence_count >= max_evidence_per_claim:
                    break

        # Evaluate all evidence in parallel
        results = await asyncio.gather(*tasks)
        # Filter out None values (failed evaluations)
        return [score for score in results if score is not None]
