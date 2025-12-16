from typing import Dict, List
import logging
import numpy as np
import asyncio

from experiments.evaluator.models import QueryGenerationMetrics
from experiments.evaluator.llm_judge.query_relevance import QueryRelevanceJudge

_logger = logging.getLogger(__name__)


class QueryGenerationEvaluator:
    """Evaluates query generation quality."""

    def __init__(self, enable_llm_judge: bool = False):
        """
        Initialize evaluator.

        Args:
            enable_llm_judge: Whether to use LLM-as-judge for query relevance
        """
        self.enable_llm_judge = enable_llm_judge
        self.judge = QueryRelevanceJudge() if enable_llm_judge else None

    def evaluate(
        self,
        outputs_data: Dict,
        llm_calls_data: List[Dict],
    ) -> QueryGenerationMetrics:
        """
        Evaluate query generation.

        Args:
            outputs_data: System outputs
            llm_calls_data: LLM calls with queries

        Returns:
            QueryGenerationMetrics
        """
        # Calculate evidence retrieval success
        claim_reports = outputs_data.get("final_output", {}).get("claim_reports", [])
        total_queries = 0
        successful_queries = 0

        for report in claim_reports:
            total_sources = report.get("total_sources", 0)
            total_queries += 2  # 2 queries per claim
            if total_sources > 0:
                successful_queries += 2

        evidence_retrieval_success = (
            successful_queries / total_queries if total_queries > 0 else 0.0
        )
        avg_sources = (
            np.mean([r.get("total_sources", 0) for r in claim_reports])
            if claim_reports
            else 0.0
        )

        # LLM-as-judge for query relevance (optional)
        query_relevance_avg = 0.0
        if self.enable_llm_judge and self.judge and llm_calls_data:
            _logger.info("  Running LLM-as-judge for query relevance...")

            # Extract query generation calls from llm_calls_data
            query_gen_calls = [
                call
                for call in llm_calls_data
                if call.get("component") == "query_generation"
            ]

            # Evaluate queries in parallel (limit to avoid excessive API calls)
            max_calls = min(len(query_gen_calls), 5)
            relevance_scores = asyncio.run(
                self._evaluate_queries_async(
                    query_gen_calls[:max_calls], max_queries_per_claim=2
                )
            )

            if relevance_scores:
                query_relevance_avg = np.mean(relevance_scores)
                _logger.info(
                    "  Evaluated %d queries in parallel, avg relevance: %.2f",
                    len(relevance_scores),
                    query_relevance_avg,
                )

        return QueryGenerationMetrics(
            query_claim_relevance_avg=query_relevance_avg,
            evidence_retrieval_success_rate=evidence_retrieval_success,
            avg_sources_per_query=avg_sources / 2 if avg_sources > 0 else 0.0,
        )

    async def _evaluate_queries_async(
        self, query_gen_calls: List[Dict], max_queries_per_claim: int = 2
    ) -> List[float]:
        """Evaluate queries in parallel using async."""

        async def evaluate_one_query(claim_text, query_text):
            try:
                score = await self.judge.judge.evaluate_async(
                    f"""Evaluate this search query:

**Claim:** {claim_text}

**Query:** {query_text}"""
                )
                return score.relevance_score
            except Exception as e:
                _logger.warning(f"    Failed to evaluate query relevance: {e}")
                return None

        # Collect all query evaluation tasks
        tasks = []
        for call in query_gen_calls:
            output = call.get("output", {})
            claim_text = output.get("original_claim", "")
            queries = output.get("queries", [])

            # Evaluate each query for this claim
            for query_obj in queries[:max_queries_per_claim]:
                query_text = query_obj.get("query", "")
                if claim_text and query_text:
                    tasks.append(evaluate_one_query(claim_text, query_text))

        # Evaluate all queries in parallel
        results = await asyncio.gather(*tasks)
        # Filter out None values (failed evaluations)
        return [score for score in results if score is not None]
