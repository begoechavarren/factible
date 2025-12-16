from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# Ground Truth Data Models


class GroundTruthClaim(BaseModel):
    """Ground truth annotation for a single claim"""

    claim_text: str
    is_checkable: bool
    importance: float  # 0.0-1.0 numerical score

    verdict: Dict  # Contains overall_stance, reasoning
    extraction_quality: Optional[Dict] = None


class VideoGroundTruth(BaseModel):
    """Complete ground truth annotation for a video"""

    video_id: str
    video_url: str
    video_title: str
    video_duration_seconds: float

    claims: List[GroundTruthClaim]


# Evaluation Results Models


class ClaimExtractionMetrics(BaseModel):
    """Metrics for claim extraction component"""

    # Basic ClaimBuster-style metrics
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1_score: float = Field(ge=0.0, le=1.0)

    true_positives: int = Field(ge=0)
    false_positives: int = Field(ge=0)
    false_negatives: int = Field(ge=0)

    # ClaimBuster ranking metrics
    mean_average_precision: float = Field(
        default=0.0, ge=0.0, le=1.0, description="MAP score for ranking quality"
    )

    # Custom importance-based metrics
    recall_at_important: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Recall for high-importance claims (>= 0.80)",
    )
    importance_weighted_coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="% of total importance covered by matched claims",
    )

    # Detailed matching info
    matched_claims: List[Dict] = Field(default_factory=list)
    missed_claims: List[str] = Field(default_factory=list)
    extra_claims: List[str] = Field(default_factory=list)

    # Importance scoring accuracy
    importance_mae: float = Field(default=0.0, ge=0.0)  # Mean absolute error

    # LLM-as-judge metrics (optional)
    claim_quality_avg: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Average claim quality score from LLM judge",
    )


class VerdictAccuracyMetrics(BaseModel):
    """Metrics for verdict accuracy"""

    overall_accuracy: float = Field(ge=0.0, le=1.0)
    stance_accuracy: float = Field(ge=0.0, le=1.0)

    correct_verdicts: int = Field(ge=0)
    total_verdicts: int = Field(ge=0)

    # Per-class metrics
    per_class_f1: Dict[str, float] = Field(default_factory=dict)
    confusion_matrix: List[List[int]] = Field(default_factory=list)

    # Error analysis
    verdict_errors: List[Dict] = Field(default_factory=list)

    # LLM-as-judge metrics (optional)
    explanation_quality_avg: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Average explanation quality score from LLM judge",
    )


class QueryGenerationMetrics(BaseModel):
    """Metrics for query generation component"""

    query_claim_relevance_avg: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Average relevance score (LLM-as-judge)",
    )
    evidence_retrieval_success_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="% of queries that retrieved evidence",
    )
    avg_sources_per_query: float = Field(default=0.0, ge=0.0)


class EvidenceSearchMetrics(BaseModel):
    """Metrics for evidence search component"""

    evidence_relevance_avg: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Average evidence relevance (LLM-as-judge)",
    )
    source_reliability_distribution: Dict[str, int] = Field(default_factory=dict)
    avg_evidence_per_claim: float = Field(default=0.0, ge=0.0)


class EndToEndMetrics(BaseModel):
    """End-to-end system performance metrics"""

    claim_coverage: float = Field(
        ge=0.0, le=1.0, description="% of ground truth claims found"
    )
    verdict_accuracy: float = Field(
        ge=0.0, le=1.0, description="% of verdicts matching ground truth"
    )
    evidence_quality_avg: float = Field(
        ge=0.0, le=1.0, description="Average evidence quality score"
    )

    total_cost_usd: float = Field(ge=0.0)
    total_time_seconds: float = Field(ge=0.0)

    # LLM-as-judge metrics
    claim_quality_avg: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Average claim quality (LLM-as-judge)"
    )
    explanation_quality_avg: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Average explanation quality (LLM-as-judge)",
    )
    user_satisfaction_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Overall user satisfaction (LLM-as-judge)",
    )


class VideoEvaluationResult(BaseModel):
    """Complete evaluation results for a single video"""

    video_id: str
    timestamp: str

    # Evaluation metadata
    evaluated_run_dir: str
    ground_truth_dir: str

    # Component metrics
    claim_extraction: ClaimExtractionMetrics
    verdict_accuracy: VerdictAccuracyMetrics
    end_to_end: EndToEndMetrics

    # Optional new metrics
    query_generation: Optional[QueryGenerationMetrics] = None
    evidence_search: Optional[EvidenceSearchMetrics] = None

    # Counts
    ground_truth_claims_count: int
    system_claims_count: int
