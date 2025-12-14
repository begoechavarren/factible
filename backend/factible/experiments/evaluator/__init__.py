"""
Modular Ground Truth Evaluation System for Factible.

This package provides a clean, modular architecture for evaluating
the fact-checking system against manually annotated ground truth data.

Main Components:
- GroundTruthEvaluator: Main orchestrator for running evaluations
- Metrics evaluators: Calculate specific performance metrics
- LLM judges: Optional LLM-as-judge evaluations
- Models: Pydantic models for data structures

Usage:
    from factible.experiments.evaluator import GroundTruthEvaluator

    evaluator = GroundTruthEvaluator(
        runs_dir=Path("data/runs/baseline"),
        ground_truth_dir=Path("data/ground_truth"),
        output_dir=Path("data/eval_results/baseline"),
        enable_llm_judge=False,  # Set to True for LLM-based metrics
    )

    results = evaluator.evaluate_all()
"""

# Main evaluator
from factible.experiments.evaluator.run_evaluator import (
    GroundTruthEvaluator,
    evaluate_runs,
)

# Ground truth management
from factible.experiments.evaluator.ground_truth import GroundTruthManager

# Claim matching utilities
from factible.experiments.evaluator.claim_matching import (
    semantic_similarity_match_claims,
    fuzzy_match_claims,
    calculate_mean_average_precision,
)

# Data models
from factible.experiments.evaluator.models import (
    # Ground truth models
    GroundTruthClaim,
    VideoGroundTruth,
    # Metric models
    ClaimExtractionMetrics,
    VerdictAccuracyMetrics,
    QueryGenerationMetrics,
    EvidenceSearchMetrics,
    EndToEndMetrics,
    # Result models
    VideoEvaluationResult,
)

# Metrics evaluators (also available via metrics subpackage)
from factible.experiments.evaluator.metrics import (
    ClaimExtractionEvaluator,
    VerdictEvaluator,
    QueryGenerationEvaluator,
    EvidenceSearchEvaluator,
    EndToEndEvaluator,
)

# LLM judges (also available via llm_judge subpackage)
from factible.experiments.evaluator.llm_judge import (
    LLMJudgeBase,
    create_simple_judge,
    ClaimQualityJudge,
    ClaimQualityScore,
    QueryRelevanceJudge,
    QueryRelevanceScore,
    EvidenceRelevanceJudge,
    EvidenceRelevanceScore,
    ExplanationQualityJudge,
    ExplanationQualityScore,
)

__all__ = [
    # Main API
    "GroundTruthEvaluator",
    "evaluate_runs",
    # Ground truth
    "GroundTruthManager",
    # Claim matching
    "semantic_similarity_match_claims",
    "fuzzy_match_claims",
    "calculate_mean_average_precision",
    # Models
    "GroundTruthClaim",
    "VideoGroundTruth",
    "ClaimExtractionMetrics",
    "VerdictAccuracyMetrics",
    "QueryGenerationMetrics",
    "EvidenceSearchMetrics",
    "EndToEndMetrics",
    "VideoEvaluationResult",
    # Metrics evaluators
    "ClaimExtractionEvaluator",
    "VerdictEvaluator",
    "QueryGenerationEvaluator",
    "EvidenceSearchEvaluator",
    "EndToEndEvaluator",
    # LLM judges
    "LLMJudgeBase",
    "create_simple_judge",
    "ClaimQualityJudge",
    "ClaimQualityScore",
    "QueryRelevanceJudge",
    "QueryRelevanceScore",
    "EvidenceRelevanceJudge",
    "EvidenceRelevanceScore",
    "ExplanationQualityJudge",
    "ExplanationQualityScore",
]
