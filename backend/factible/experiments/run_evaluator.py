"""
Ground Truth Evaluation Module

Evaluates fact-checking system performance against manually annotated ground truth data.

CLI Usage:
    # Evaluate experiment runs against ground truth
    uv run factible-experiments evaluate \\
        --runs-dir factible/experiments/data/runs/20251213_114139_baseline

    # With custom ground truth location
    uv run factible-experiments evaluate \\
        --runs-dir factible/experiments/data/runs/20251213_114139_baseline \\
        --ground-truth-dir factible/experiments/data/ground_truth

Python Usage:
    from factible.experiments.run_evaluator import evaluate_runs

    evaluate_runs(
        runs_dir="factible/experiments/data/runs/20251213_114139_baseline",
        ground_truth_dir="factible/experiments/data/ground_truth"
    )
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
from difflib import SequenceMatcher
import json
from datetime import datetime

from pydantic import BaseModel, Field
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    accuracy_score,
)


# ============================================================================
# Ground Truth Data Models
# ============================================================================


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


# ============================================================================
# Evaluation Results Models
# ============================================================================


class ClaimExtractionMetrics(BaseModel):
    """Metrics for claim extraction component"""

    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1_score: float = Field(ge=0.0, le=1.0)

    true_positives: int = Field(ge=0)
    false_positives: int = Field(ge=0)
    false_negatives: int = Field(ge=0)

    # Detailed matching info
    matched_claims: List[Dict] = Field(default_factory=list)
    missed_claims: List[str] = Field(default_factory=list)
    extra_claims: List[str] = Field(default_factory=list)

    # Importance scoring accuracy
    importance_mae: float = Field(default=0.0, ge=0.0)  # Mean absolute error


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


class VideoEvaluationResult(BaseModel):
    """Complete evaluation results for a single video"""

    video_id: str
    timestamp: str

    # Evaluation metadata
    evaluated_run_dir: str
    ground_truth_dir: str

    claim_extraction: ClaimExtractionMetrics
    verdict_accuracy: VerdictAccuracyMetrics
    end_to_end: EndToEndMetrics

    ground_truth_claims_count: int
    system_claims_count: int


# ============================================================================
# Ground Truth Manager
# ============================================================================


class GroundTruthManager:
    """Manages loading and caching of ground truth annotations"""

    def __init__(self, ground_truth_dir: Path):
        self.gt_dir = Path(ground_truth_dir)
        self.cache: Dict[str, VideoGroundTruth] = {}

        if not self.gt_dir.exists():
            raise FileNotFoundError(
                f"Ground truth directory does not exist: {self.gt_dir}\n"
                f"Expected ground truth YAML files in this location."
            )

    def load(self, video_id: str) -> VideoGroundTruth:
        """Load ground truth for a video"""
        if video_id in self.cache:
            return self.cache[video_id]

        path = self.gt_dir / f"{video_id}.yaml"
        if not path.exists():
            raise FileNotFoundError(
                f"Ground truth file not found: {path}\n"
                f"Create it using the ground truth template as reference."
            )

        with open(path) as f:
            data = yaml.safe_load(f)

        gt = VideoGroundTruth(**data)
        self.cache[video_id] = gt
        return gt

    def list_available(self) -> List[str]:
        """List all available ground truth video IDs"""
        yaml_files = self.gt_dir.glob("*.yaml")
        return [f.stem for f in yaml_files if f.stem != "TEMPLATE_video"]


# ============================================================================
# Claim Matching Utilities
# ============================================================================


def fuzzy_match_claims(
    gt_claims: List[GroundTruthClaim],
    system_claims: List,
    threshold: float = 0.7,
) -> Dict[str, List]:
    """
    Match system-extracted claims to ground truth claims using fuzzy string matching.

    Returns:
        {
            "true_positives": [(gt_claim, system_claim), ...],
            "false_positives": [system_claim, ...],  # Extra claims
            "false_negatives": [gt_claim, ...],      # Missed claims
        }
    """
    matched_pairs = []
    unmatched_gt = list(gt_claims)
    unmatched_system = list(system_claims)

    # Find best matches
    for gt_claim in gt_claims:
        best_match = None
        best_score = threshold

        for sys_claim in unmatched_system:
            score = SequenceMatcher(
                None, gt_claim.claim_text.lower(), sys_claim.text.lower()
            ).ratio()

            if score > best_score:
                best_match = sys_claim
                best_score = score

        if best_match:
            matched_pairs.append((gt_claim, best_match))
            unmatched_gt.remove(gt_claim)
            unmatched_system.remove(best_match)

    return {
        "true_positives": matched_pairs,
        "false_positives": unmatched_system,
        "false_negatives": unmatched_gt,
    }


# ============================================================================
# Main Evaluator
# ============================================================================


class GroundTruthEvaluator:
    """Comprehensive evaluator for multi-agent fact-checking system"""

    def __init__(
        self,
        runs_dir: Path,
        ground_truth_dir: Path,
        output_dir: Path,
    ):
        self.runs_dir = Path(runs_dir)
        self.gt_mgr = GroundTruthManager(ground_truth_dir)
        self.ground_truth_dir = Path(ground_truth_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def find_matching_videos(self) -> List[str]:
        """Find video IDs that exist in both runs and ground truth"""
        gt_videos = set(self.gt_mgr.list_available())

        # Find all video IDs in runs directory
        # Run directory format: YYYYMMDD_HHMMSS_{video_id}
        # We need to extract {video_id} which may contain underscores
        run_videos = set()
        for run_dir in self.runs_dir.iterdir():
            if not run_dir.is_dir():
                continue
            # Skip first two parts (date and time), rest is video_id
            parts = run_dir.name.split("_")
            if len(parts) >= 3:
                video_id = "_".join(parts[2:])  # Join remaining parts
                run_videos.add(video_id)

        matching = gt_videos & run_videos
        return sorted(list(matching))

    def evaluate_video(
        self, video_id: str, run_dir: Optional[Path] = None
    ) -> VideoEvaluationResult:
        """
        Evaluate system performance on a single video.

        Args:
            video_id: ID of video to evaluate
            run_dir: Path to specific run directory. If None, searches for matching run.

        Returns:
            VideoEvaluationResult with all metrics
        """
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {video_id}")
        print(f"{'=' * 60}")

        # Load ground truth
        gt = self.gt_mgr.load(video_id)
        print(f"Loaded ground truth: {len(gt.claims)} claims")

        # Load run outputs
        if run_dir is None:
            run_dir = self._find_run_dir(video_id)

        outputs_data, metrics_data = self._load_run_outputs(run_dir)

        print(f"Loaded run from: {run_dir}")
        print(
            f"System extracted: {len(outputs_data['extracted_claims']['claims'])} claims"
        )

        # Evaluate claim extraction
        claim_metrics = self._evaluate_claim_extraction(
            gt.claims, outputs_data["extracted_claims"]["claims"]
        )
        print(
            f"Claim Extraction: P={claim_metrics.precision:.2f} "
            f"R={claim_metrics.recall:.2f} F1={claim_metrics.f1_score:.2f}"
        )
        print(f"Importance MAE: {claim_metrics.importance_mae:.3f}")

        # Evaluate verdict accuracy
        claim_reports = outputs_data["final_output"]["claim_reports"]
        verdict_metrics = self._evaluate_verdicts(gt.claims, claim_reports)
        print(f"Stance Accuracy: {verdict_metrics.stance_accuracy:.2%}")

        # Calculate end-to-end metrics
        e2e_metrics = self._calculate_e2e_metrics(
            gt.claims, outputs_data, metrics_data, claim_metrics, verdict_metrics
        )
        print(
            f"End-to-End: Coverage={e2e_metrics.claim_coverage:.2%} "
            f"Accuracy={e2e_metrics.verdict_accuracy:.2%}"
        )
        print(f"Avg Evidence Quality: {e2e_metrics.evidence_quality_avg:.2f}")
        print(f"Total Cost: ${e2e_metrics.total_cost_usd:.4f}")
        print(f"Total Time: {e2e_metrics.total_time_seconds:.1f}s")

        result = VideoEvaluationResult(
            video_id=video_id,
            timestamp=datetime.now().isoformat(),
            evaluated_run_dir=str(self.runs_dir),
            ground_truth_dir=str(self.ground_truth_dir),
            claim_extraction=claim_metrics,
            verdict_accuracy=verdict_metrics,
            end_to_end=e2e_metrics,
            ground_truth_claims_count=len(gt.claims),
            system_claims_count=len(outputs_data["extracted_claims"]["claims"]),
        )

        return result

    def _evaluate_claim_extraction(
        self, gt_claims: List[GroundTruthClaim], extracted_claims_data: List[Dict]
    ) -> ClaimExtractionMetrics:
        """Evaluate claim extraction precision/recall"""

        # Convert dict to simple objects for matching
        class SimpleClaim:
            def __init__(self, data):
                self.text = data["text"]
                self.importance = data.get("importance", 0.0)

        system_claims = [SimpleClaim(c) for c in extracted_claims_data]

        matches = fuzzy_match_claims(gt_claims, system_claims)

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

        # Calculate importance scoring accuracy (MAE)
        importance_errors = []
        for gt_claim, sys_claim in matches["true_positives"]:
            gt_imp = gt_claim.importance  # 0.0-1.0
            sys_imp = sys_claim.importance  # 0.0-1.0
            importance_errors.append(abs(gt_imp - sys_imp))

        importance_mae = np.mean(importance_errors) if importance_errors else 0.0

        return ClaimExtractionMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            importance_mae=importance_mae,
            matched_claims=[
                {
                    "gt_text": gt.claim_text,
                    "sys_text": sys.text,
                    "gt_importance": gt.importance,
                    "sys_importance": sys.importance,
                    "importance_error": abs(gt.importance - sys.importance),
                }
                for gt, sys in matches["true_positives"]
            ],
            missed_claims=[gt.claim_text for gt in matches["false_negatives"]],
            extra_claims=[sys.text for sys in matches["false_positives"]],
        )

    def _evaluate_verdicts(
        self,
        gt_claims: List[GroundTruthClaim],
        claim_reports_data: List[Dict],
    ) -> VerdictAccuracyMetrics:
        """Evaluate verdict classification accuracy"""

        # Match ground truth claims to system reports
        y_true = []
        y_pred = []
        errors = []

        for gt_claim in gt_claims:
            # Find matching report by fuzzy text matching
            best_match = None
            best_score = 0.7

            for report in claim_reports_data:
                score = SequenceMatcher(
                    None, gt_claim.claim_text.lower(), report["claim_text"].lower()
                ).ratio()

                if score > best_score:
                    best_match = report
                    best_score = score

            if best_match:
                # GT uses overall_stance (SUPPORTS/REFUTES/MIXED/UNCLEAR)
                # System uses overall_stance (supports/refutes/mixed/unclear) - lowercase
                gt_stance = gt_claim.verdict.get("overall_stance", "UNCLEAR").upper()
                sys_stance = best_match["overall_stance"].upper()

                y_true.append(gt_stance)
                y_pred.append(sys_stance)

                if gt_stance != sys_stance:
                    errors.append(
                        {
                            "claim": gt_claim.claim_text,
                            "expected": gt_stance,
                            "predicted": sys_stance,
                            "confidence": best_match["verdict_confidence"],
                        }
                    )

        if not y_true:
            # No matches found
            return VerdictAccuracyMetrics(
                overall_accuracy=0.0,
                stance_accuracy=0.0,
                correct_verdicts=0,
                total_verdicts=len(gt_claims),
                verdict_errors=errors,
            )

        # Calculate stance accuracy (direct comparison)
        stance_accuracy = accuracy_score(y_true, y_pred)

        # Confusion matrix using actual labels
        labels = ["SUPPORTS", "REFUTES", "MIXED", "UNCLEAR"]
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        # Per-class F1
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        )

        per_class_f1 = {
            label: f1_score
            for label, f1_score in zip(labels, f1)
            if label in set(y_true) or label in set(y_pred)
        }

        return VerdictAccuracyMetrics(
            overall_accuracy=stance_accuracy,  # Same as stance_accuracy now
            stance_accuracy=stance_accuracy,
            correct_verdicts=sum(1 for t, p in zip(y_true, y_pred) if t == p),
            total_verdicts=len(y_true),
            per_class_f1=per_class_f1,
            confusion_matrix=cm.tolist(),
            verdict_errors=errors,
        )

    def _calculate_e2e_metrics(
        self,
        gt_claims: List[GroundTruthClaim],
        outputs_data: Dict,
        metrics_data: Dict,
        claim_metrics: ClaimExtractionMetrics,
        verdict_metrics: VerdictAccuracyMetrics,
    ) -> EndToEndMetrics:
        """Calculate end-to-end system metrics"""

        # Claim coverage = recall from claim extraction
        claim_coverage = claim_metrics.recall

        # Verdict accuracy from verdict evaluation
        verdict_accuracy = verdict_metrics.stance_accuracy

        # Average evidence quality
        claim_reports = outputs_data["final_output"]["claim_reports"]
        evidence_quality_scores = [r["evidence_quality_score"] for r in claim_reports]
        evidence_quality_avg = (
            np.mean(evidence_quality_scores) if evidence_quality_scores else 0.0
        )

        # Cost and time from metrics.json
        total_cost_usd = metrics_data.get("llm", {}).get("total_cost_usd", 0.0)
        total_time_seconds = metrics_data.get("timing", {}).get("total_seconds", 0.0)

        return EndToEndMetrics(
            claim_coverage=claim_coverage,
            verdict_accuracy=verdict_accuracy,
            evidence_quality_avg=evidence_quality_avg,
            total_cost_usd=total_cost_usd,
            total_time_seconds=total_time_seconds,
        )

    def _find_run_dir(self, video_id: str) -> Path:
        """Find run directory for a video ID"""
        # Search directly in runs_dir for directories containing video_id
        for run_dir in self.runs_dir.iterdir():
            if not run_dir.is_dir():
                continue

            # Check if video_id is in the run directory name
            if video_id in run_dir.name:
                return run_dir

        raise FileNotFoundError(
            f"Could not find run directory for video_id: {video_id}\n"
            f"Searched in: {self.runs_dir}\n"
            f"Available runs: {[d.name for d in self.runs_dir.iterdir() if d.is_dir()]}"
        )

    def _load_run_outputs(self, run_dir: Path) -> Tuple[Dict, Dict]:
        """Load outputs.json and metrics.json from a run directory"""
        outputs_path = run_dir / "outputs.json"
        metrics_path = run_dir / "metrics.json"

        if not outputs_path.exists():
            raise FileNotFoundError(f"outputs.json not found in {run_dir}")

        if not metrics_path.exists():
            raise FileNotFoundError(f"metrics.json not found in {run_dir}")

        with open(outputs_path) as f:
            outputs_data = json.load(f)

        with open(metrics_path) as f:
            metrics_data = json.load(f)

        return outputs_data, metrics_data

    def _save_result(self, video_id: str, result: VideoEvaluationResult):
        """Save evaluation result to JSON with datetime_videoid format"""
        # Format: {datetime}_{video_id}.json
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp_str}_{video_id}.json"
        output_path = self.output_dir / filename

        with open(output_path, "w") as f:
            json.dump(result.model_dump(), f, indent=2)

        print(f"\nSaved results to: {output_path}")

    def evaluate_all(self) -> List[VideoEvaluationResult]:
        """Evaluate all videos that exist in both runs and ground truth"""
        video_ids = self.find_matching_videos()

        if not video_ids:
            print("\n⚠️  No matching videos found between runs and ground truth!")
            print(f"Runs dir: {self.runs_dir}")
            print(f"GT dir: {self.gt_mgr.gt_dir}")
            return []

        print(f"\nFound {len(video_ids)} matching videos:")
        for vid in video_ids:
            print(f"  - {vid}")

        results = []
        for i, video_id in enumerate(video_ids, 1):
            print(f"\n\n[{i}/{len(video_ids)}] Evaluating {video_id}...")
            try:
                result = self.evaluate_video(video_id)
                self._save_result(video_id, result)
                results.append(result)
            except Exception as e:
                print(f"ERROR: Failed to evaluate {video_id}: {e}")
                import traceback

                traceback.print_exc()
                continue

        # Generate aggregate report
        if results:
            self.generate_aggregate_report(results)

        return results

    def generate_aggregate_report(self, results: List[VideoEvaluationResult]):
        """Generate aggregate statistics across all videos"""
        print(f"\n\n{'=' * 60}")
        print("AGGREGATE RESULTS")
        print(f"{'=' * 60}\n")

        # Claim extraction metrics
        avg_precision = np.mean([r.claim_extraction.precision for r in results])
        avg_recall = np.mean([r.claim_extraction.recall for r in results])
        avg_f1 = np.mean([r.claim_extraction.f1_score for r in results])

        print("Claim Extraction:")
        print(
            f"  Precision: {avg_precision:.2%} (σ={np.std([r.claim_extraction.precision for r in results]):.2%})"
        )
        print(
            f"  Recall:    {avg_recall:.2%} (σ={np.std([r.claim_extraction.recall for r in results]):.2%})"
        )
        print(
            f"  F1 Score:  {avg_f1:.2%} (σ={np.std([r.claim_extraction.f1_score for r in results]):.2%})"
        )

        # Verdict accuracy
        avg_verdict_acc = np.mean(
            [r.verdict_accuracy.overall_accuracy for r in results]
        )
        print(
            f"\nVerdict Accuracy: {avg_verdict_acc:.2%} (σ={np.std([r.verdict_accuracy.overall_accuracy for r in results]):.2%})"
        )

        # End-to-end
        avg_coverage = np.mean([r.end_to_end.claim_coverage for r in results])
        avg_e2e_acc = np.mean([r.end_to_end.verdict_accuracy for r in results])

        print("\nEnd-to-End:")
        print(f"  Claim Coverage:   {avg_coverage:.2%}")
        print(f"  Verdict Accuracy: {avg_e2e_acc:.2%}")

        # Save aggregate report
        report_path = self.output_dir / "aggregate_report.json"
        with open(report_path, "w") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "n_videos": len(results),
                    "claim_extraction": {
                        "precision_mean": avg_precision,
                        "recall_mean": avg_recall,
                        "f1_mean": avg_f1,
                    },
                    "verdict_accuracy_mean": avg_verdict_acc,
                    "end_to_end": {
                        "coverage_mean": avg_coverage,
                        "accuracy_mean": avg_e2e_acc,
                    },
                },
                f,
                indent=2,
            )

        print(f"\nAggregate report saved to: {report_path}")


# ============================================================================
# Public API
# ============================================================================


def evaluate_runs(
    runs_dir: str,
    ground_truth_dir: str,
) -> List[VideoEvaluationResult]:
    """
    Evaluate experiment runs against ground truth.

    Args:
        runs_dir: Path to runs directory (e.g., "runs/20251213_114139_baseline")
        ground_truth_dir: Path to ground truth directory

    Returns:
        List of evaluation results for each video
    """
    runs_path = Path(runs_dir)

    # Extract run name from runs_dir (e.g., "20251213_114139_baseline")
    run_name = runs_path.name

    # Create timestamp for this evaluation
    eval_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output directory: data/eval_results/{run_name}/{timestamp}/
    base_eval_dir = Path("factible/experiments/data/eval_results")
    output_dir = base_eval_dir / run_name / eval_timestamp

    print(f"\n{'=' * 60}")
    print(f"Evaluation: {run_name}/{eval_timestamp}")
    print(f"{'=' * 60}")
    print(f"Runs dir: {runs_dir}")
    print(f"Ground truth dir: {ground_truth_dir}")
    print(f"Output dir: {output_dir}")

    evaluator = GroundTruthEvaluator(
        runs_dir=runs_path,
        ground_truth_dir=Path(ground_truth_dir),
        output_dir=output_dir,
    )

    return evaluator.evaluate_all()
