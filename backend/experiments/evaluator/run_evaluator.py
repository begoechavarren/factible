from pathlib import Path
from typing import List
import json
from datetime import datetime
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from experiments.evaluator.models import VideoEvaluationResult
from experiments.evaluator.ground_truth import GroundTruthManager
from experiments.evaluator.metrics.claim_extraction import ClaimExtractionEvaluator
from experiments.evaluator.metrics.verdict import VerdictEvaluator
from experiments.evaluator.metrics.query_generation import QueryGenerationEvaluator
from experiments.evaluator.metrics.evidence_search import EvidenceSearchEvaluator
from experiments.evaluator.metrics.end_to_end import EndToEndEvaluator

_logger = logging.getLogger(__name__)


class GroundTruthEvaluator:
    """
    Main evaluator that orchestrates all evaluation components.

    This is the new modular version - clean, maintainable, and extensible.
    """

    def __init__(
        self,
        runs_dir: Path,
        ground_truth_dir: Path,
        output_dir: Path,
        enable_llm_judge: bool = False,
        max_workers: int = 4,
    ):
        """
        Initialize evaluator with all components.

        Args:
            runs_dir: Directory with experiment runs
            ground_truth_dir: Directory with ground truth YAML files
            output_dir: Where to save evaluation results
            enable_llm_judge: Whether to use LLM-as-judge metrics (costs API calls)
            max_workers: Number of parallel workers for evaluation (default: 4)
        """
        self.runs_dir = Path(runs_dir)
        self.ground_truth_dir = Path(ground_truth_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers

        # Initialize components
        self.gt_mgr = GroundTruthManager(ground_truth_dir)

        # Load sentence-transformer model once for all evaluations
        try:
            from sentence_transformers import SentenceTransformer

            self.similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            self.similarity_model = None

        # Initialize metric evaluators
        self.claim_extractor_eval = ClaimExtractionEvaluator(
            self.similarity_model, enable_llm_judge
        )
        self.verdict_eval = VerdictEvaluator(enable_llm_judge)
        self.query_eval = QueryGenerationEvaluator(enable_llm_judge)
        self.evidence_eval = EvidenceSearchEvaluator(enable_llm_judge)
        self.end_to_end_eval = EndToEndEvaluator()

    def find_matching_videos(self) -> List[str]:
        """Find video IDs that exist in both runs and ground truth."""
        gt_videos = set(self.gt_mgr.list_available())

        # Find all video IDs in runs directory
        run_videos = set()
        for run_dir in self.runs_dir.iterdir():
            if not run_dir.is_dir():
                continue
            # Extract video_id from directory name: YYYYMMDD_HHMMSS_{video_id}
            parts = run_dir.name.split("_")
            if len(parts) >= 3:
                video_id = "_".join(parts[2:])
                run_videos.add(video_id)

        matching = gt_videos & run_videos
        return sorted(list(matching))

    def _find_run_dir(self, video_id: str) -> Path:
        """Find the run directory for a video."""
        for run_dir in self.runs_dir.iterdir():
            if not run_dir.is_dir():
                continue
            if run_dir.name.endswith(video_id):
                return run_dir
        raise FileNotFoundError(f"No run directory found for video: {video_id}")

    def evaluate_video(self, video_id: str) -> VideoEvaluationResult:
        """
        Evaluate system performance on a single video.

        Args:
            video_id: Video ID to evaluate

        Returns:
            VideoEvaluationResult with all metrics
        """
        _logger.info(f"\n{'=' * 60}")
        _logger.info(f"Evaluating: {video_id}")
        _logger.info(f"{'=' * 60}")

        # Load ground truth
        gt = self.gt_mgr.load(video_id)
        _logger.info(f"Loaded ground truth: {len(gt.claims)} claims")

        # Find and load run data
        run_dir = self._find_run_dir(video_id)
        _logger.info(f"Loaded run from: {run_dir}")

        # Load run outputs
        with open(run_dir / "outputs.json") as f:
            outputs_data = json.load(f)

        with open(run_dir / "metrics.json") as f:
            metrics_data = json.load(f)

        with open(run_dir / "config.json") as f:
            config_data = json.load(f)

        # Load LLM calls data (for query evaluation)
        llm_calls_data = []
        llm_calls_path = run_dir / "llm_calls.json"
        if llm_calls_path.exists():
            with open(llm_calls_path) as f:
                llm_calls_data = json.load(f)

        _logger.info(f"System config: max_claims={config_data.get('max_claims')}")

        # Extract system claims
        extracted_claims_data = outputs_data["extracted_claims"]["claims"]
        _logger.info(f"System extracted: {len(extracted_claims_data)} claims")
        _logger.info(f"Evaluating against: {len(gt.claims)} ground truth claims (ALL)")

        # 1. Evaluate claim extraction
        claim_metrics = self.claim_extractor_eval.evaluate(
            gt.claims, extracted_claims_data
        )
        _logger.info(
            "Claim Extraction: P=%.2f R=%.2f F1=%.2f",
            claim_metrics.precision,
            claim_metrics.recall,
            claim_metrics.f1_score,
        )
        _logger.info("MAP: %.3f", claim_metrics.mean_average_precision)
        _logger.info(
            "Recall@Important: %.2f%%", claim_metrics.recall_at_important * 100
        )
        _logger.info(
            "Importance Coverage: %.2f%%",
            claim_metrics.importance_weighted_coverage * 100,
        )
        _logger.info("Importance MAE: %.3f", claim_metrics.importance_mae)

        # 2. Evaluate verdicts
        # Handle case where no claims were extracted (no final_output)
        if (
            "final_output" in outputs_data
            and "claim_reports" in outputs_data["final_output"]
        ):
            claim_reports = outputs_data["final_output"]["claim_reports"]
        else:
            claim_reports = []

        verdict_metrics = self.verdict_eval.evaluate(
            gt.claims, claim_reports, claim_metrics.matched_claims
        )
        _logger.info("Stance Accuracy: %.2f%%", verdict_metrics.stance_accuracy * 100)

        # 3. Evaluate query generation (optional with LLM judge)
        query_metrics = self.query_eval.evaluate(outputs_data, llm_calls_data)

        # 4. Evaluate evidence search (optional with LLM judge)
        evidence_metrics = self.evidence_eval.evaluate(outputs_data)

        # 5. Calculate end-to-end metrics
        end_to_end_metrics = self.end_to_end_eval.evaluate(
            claim_coverage=claim_metrics.recall,
            verdict_accuracy=verdict_metrics.overall_accuracy,
            metrics_data=metrics_data,
            outputs_data=outputs_data,
        )
        _logger.info(
            "End-to-End: Coverage=%.2f%% Accuracy=%.2f%%",
            end_to_end_metrics.claim_coverage * 100,
            end_to_end_metrics.verdict_accuracy * 100,
        )
        _logger.info(
            "Avg Evidence Quality: %.2f", end_to_end_metrics.evidence_quality_avg
        )
        _logger.info("Total Cost: $%.4f", end_to_end_metrics.total_cost_usd)
        _logger.info("Total Time: %.1fs", end_to_end_metrics.total_time_seconds)

        # Create result
        result = VideoEvaluationResult(
            video_id=video_id,
            timestamp=datetime.now().isoformat(),
            evaluated_run_dir=str(self.runs_dir),
            ground_truth_dir=str(self.ground_truth_dir),
            claim_extraction=claim_metrics,
            verdict_accuracy=verdict_metrics,
            end_to_end=end_to_end_metrics,
            query_generation=query_metrics,
            evidence_search=evidence_metrics,
            ground_truth_claims_count=len(gt.claims),
            system_claims_count=len(extracted_claims_data),
        )

        # Save individual result
        result_path = (
            self.output_dir
            / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{video_id}.json"
        )
        with open(result_path, "w") as f:
            json.dump(result.model_dump(), f, indent=2)

        _logger.info(f"\nSaved results to: {result_path}")

        return result

    def evaluate_all(self) -> List[VideoEvaluationResult]:
        """
        Evaluate all matching videos in parallel.

        Returns:
            List of VideoEvaluationResult for all videos
        """
        matching_videos = self.find_matching_videos()

        _logger.info(f"\n{'=' * 60}")
        _logger.info(
            f"Evaluation: {self.runs_dir.name}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        _logger.info(f"{'=' * 60}")
        _logger.info(f"Runs dir: {self.runs_dir}")
        _logger.info(f"Ground truth dir: {self.ground_truth_dir}")
        _logger.info(f"Output dir: {self.output_dir}")
        _logger.info(f"Parallelization: {self.max_workers} workers")
        _logger.info(f"\nFound {len(matching_videos)} matching videos:")
        for vid in matching_videos:
            _logger.info(f"  - {vid}")

        # Parallel evaluation using ThreadPoolExecutor
        results = []
        completed = 0
        total = len(matching_videos)

        _logger.info(
            "\nEvaluating %d videos in PARALLEL with %d workers",
            total,
            self.max_workers,
        )
        _logger.info(f"{'-' * 60}")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all video evaluation tasks
            future_to_video = {
                executor.submit(self.evaluate_video, video_id): video_id
                for video_id in matching_videos
            }

            # Process results as they complete
            for future in as_completed(future_to_video):
                video_id = future_to_video[future]
                completed += 1
                try:
                    result = future.result()
                    results.append(result)
                    _logger.info(f"[{completed}/{total}] Completed: {video_id}")
                except Exception as exc:
                    _logger.error(f"[{completed}/{total}] Failed: {video_id} - {exc}")

        _logger.info(f"\nCompleted all {total} evaluations")
        _logger.info(f"{'-' * 60}")

        # Sort results by video_id for consistent ordering
        results.sort(key=lambda r: r.video_id)

        # Generate aggregate report
        self.generate_aggregate_report(results)

        return results

    def generate_aggregate_report(self, results: List[VideoEvaluationResult]):
        """Generate aggregate statistics across all videos."""
        _logger.info(f"\n\n{'=' * 60}")
        _logger.info("AGGREGATE RESULTS")
        _logger.info(f"{'=' * 60}\n")

        # Extract max_claims from first run's config
        max_claims = None
        try:
            first_video_id = results[0].video_id
            run_dir = self._find_run_dir(first_video_id)
            with open(run_dir / "config.json") as f:
                config_data = json.load(f)
                max_claims = config_data.get("max_claims")
        except Exception:
            pass

        # Claim extraction metrics
        avg_precision = np.mean([r.claim_extraction.precision for r in results])
        avg_recall = np.mean([r.claim_extraction.recall for r in results])
        avg_f1 = np.mean([r.claim_extraction.f1_score for r in results])
        avg_map = np.mean([r.claim_extraction.mean_average_precision for r in results])
        avg_recall_important = np.mean(
            [r.claim_extraction.recall_at_important for r in results]
        )
        avg_importance_coverage = np.mean(
            [r.claim_extraction.importance_weighted_coverage for r in results]
        )

        _logger.info("Claim Extraction:")
        _logger.info(
            "  Precision: %.2f%% (σ=%.2f%%)",
            avg_precision * 100,
            np.std([r.claim_extraction.precision for r in results]) * 100,
        )
        _logger.info(
            "  Recall:    %.2f%% (σ=%.2f%%)",
            avg_recall * 100,
            np.std([r.claim_extraction.recall for r in results]) * 100,
        )
        _logger.info(
            "  F1 Score:  %.2f%% (σ=%.2f%%)",
            avg_f1 * 100,
            np.std([r.claim_extraction.f1_score for r in results]) * 100,
        )
        _logger.info("\n  MAP (ClaimBuster): %.3f", avg_map)
        _logger.info("  Recall@Important:  %.2f%%", avg_recall_important * 100)
        _logger.info("  Importance Coverage: %.2f%%", avg_importance_coverage * 100)

        # Verdict accuracy
        avg_verdict_acc = np.mean(
            [r.verdict_accuracy.overall_accuracy for r in results]
        )
        avg_stance_acc = np.mean([r.verdict_accuracy.stance_accuracy for r in results])
        _logger.info(
            "\nVerdict Accuracy: %.2f%% (σ=%.2f%%)",
            avg_verdict_acc * 100,
            np.std([r.verdict_accuracy.overall_accuracy for r in results]) * 100,
        )
        _logger.info(
            "  Stance Accuracy: %.2f%% (σ=%.2f%%)",
            avg_stance_acc * 100,
            np.std([r.verdict_accuracy.stance_accuracy for r in results]) * 100,
        )

        # End-to-end
        avg_coverage = np.mean([r.end_to_end.claim_coverage for r in results])
        avg_e2e_acc = np.mean([r.end_to_end.verdict_accuracy for r in results])
        avg_latency = np.mean([r.end_to_end.total_time_seconds for r in results])
        std_latency = np.std([r.end_to_end.total_time_seconds for r in results])
        total_latency = np.sum([r.end_to_end.total_time_seconds for r in results])
        avg_cost = np.mean([r.end_to_end.total_cost_usd for r in results])
        total_cost = np.sum([r.end_to_end.total_cost_usd for r in results])

        _logger.info("\nEnd-to-End:")
        _logger.info("  Claim Coverage:   %.2f%%", avg_coverage * 100)
        _logger.info("  Verdict Accuracy: %.2f%%", avg_e2e_acc * 100)
        _logger.info("  Avg Latency:      %.2fs (σ=%.2fs)", avg_latency, std_latency)
        _logger.info("  Total Latency:    %.2fs", total_latency)
        _logger.info("  Avg Cost:         $%.4f", avg_cost)
        _logger.info("  Total Cost:       $%.4f", total_cost)

        # Build aggregate report
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "n_videos": len(results),
            "description": {
                "precision_at_k": "Of k claims extracted, % that match ground truth (any GT claim)",
                "recall": "Of all GT claims, % that were matched by extracted claims",
                "f1": "Harmonic mean of precision and recall",
                "map": "Mean Average Precision - ranking quality (ClaimBuster metric)",
                "recall_at_important": "Recall for GT claims with importance >= 0.80",
                "importance_weighted_coverage": "% of total GT importance covered by matches",
                "verdict_overall_accuracy": "% of verdicts that match ground truth (both stance and factual assessment)",
                "verdict_stance_accuracy": "% of verdicts where stance classification matches ground truth",
                "latency": "Processing time in seconds per video",
                "cost": "LLM API cost in USD per video",
            },
            "claim_extraction": {
                "precision_at_k_mean": avg_precision,
                "precision_at_k_std": np.std(
                    [r.claim_extraction.precision for r in results]
                ),
                "recall_mean": avg_recall,
                "recall_std": np.std([r.claim_extraction.recall for r in results]),
                "f1_mean": avg_f1,
                "f1_std": np.std([r.claim_extraction.f1_score for r in results]),
                "map_mean": avg_map,
                "map_std": np.std(
                    [r.claim_extraction.mean_average_precision for r in results]
                ),
                "recall_at_important_mean": avg_recall_important,
                "recall_at_important_std": np.std(
                    [r.claim_extraction.recall_at_important for r in results]
                ),
                "importance_weighted_coverage_mean": avg_importance_coverage,
                "importance_weighted_coverage_std": np.std(
                    [r.claim_extraction.importance_weighted_coverage for r in results]
                ),
            },
            "verdict": {
                "overall_accuracy_mean": avg_verdict_acc,
                "overall_accuracy_std": np.std(
                    [r.verdict_accuracy.overall_accuracy for r in results]
                ),
                "stance_accuracy_mean": avg_stance_acc,
                "stance_accuracy_std": np.std(
                    [r.verdict_accuracy.stance_accuracy for r in results]
                ),
            },
            "end_to_end": {
                "coverage_mean": avg_coverage,
                "coverage_std": np.std([r.end_to_end.claim_coverage for r in results]),
                "accuracy_mean": avg_e2e_acc,
                "accuracy_std": np.std(
                    [r.end_to_end.verdict_accuracy for r in results]
                ),
                "latency_mean_seconds": avg_latency,
                "latency_std_seconds": std_latency,
                "latency_total_seconds": total_latency,
                "cost_mean_usd": avg_cost,
                "cost_total_usd": total_cost,
            },
        }

        if max_claims is not None:
            report_data["max_claims"] = max_claims

        # Save aggregate report
        report_path = self.output_dir / "aggregate_report.json"
        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=2)

        _logger.info(f"\nAggregate report saved to: {report_path}")


def evaluate_runs(
    runs_dir: str,
    ground_truth_dir: str,
    enable_llm_judge: bool = False,
    max_workers: int = 4,
) -> List[VideoEvaluationResult]:
    """
    Evaluate experiment runs against ground truth.

    Args:
        runs_dir: Path to runs directory
        ground_truth_dir: Path to ground truth directory
        enable_llm_judge: Whether to use LLM-as-judge metrics (costs API calls)
        max_workers: Number of parallel workers (default: 4)

    Returns:
        List of evaluation results for each video
    """
    runs_path = Path(runs_dir)
    gt_path = Path(ground_truth_dir)

    # Create output directory based on run directory name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = runs_path.parent.parent / "eval_results" / runs_path.name / timestamp

    evaluator = GroundTruthEvaluator(
        runs_dir=runs_path,
        ground_truth_dir=gt_path,
        output_dir=output_dir,
        enable_llm_judge=enable_llm_judge,
        max_workers=max_workers,
    )

    return evaluator.evaluate_all()
