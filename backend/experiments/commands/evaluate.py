from datetime import datetime
from pathlib import Path
from typing import Annotated, List

import typer

from experiments.evaluator.evaluator import GroundTruthEvaluator
from experiments.evaluator.models import VideoEvaluationResult


def evaluate_command(
    runs_dir: Annotated[
        str,
        typer.Option(help="Path to runs directory"),
    ],
    ground_truth_dir: Annotated[
        str,
        typer.Option(help="Path to ground truth directory"),
    ] = "experiments/data/ground_truth",
    enable_llm_judge: Annotated[
        bool,
        typer.Option(help="Enable LLM-as-judge evaluation (costs API calls)"),
    ] = False,
    max_workers: Annotated[
        int,
        typer.Option(help="Number of parallel workers for evaluation"),
    ] = 4,
) -> List[VideoEvaluationResult]:
    """
    Evaluate experiment runs against manually annotated ground truth.

    Examples:
        factible-experiments evaluate --runs-dir experiments/data/runs/my_experiment
    """
    runs_path = Path(runs_dir).resolve()
    gt_path = Path(ground_truth_dir)

    # Determine output directory by mirroring the runs path structure under eval_results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Find the /runs directory in the path and compute relative path from there
    runs_base = Path("experiments/data/runs").resolve()
    try:
        relative_path = runs_path.relative_to(runs_base)
        output_dir = Path("experiments/data/eval_results") / relative_path / timestamp
    except ValueError:
        # Fallback if runs_dir is not under experiments/data/runs
        output_dir = runs_path.parent / "eval_results" / runs_path.name / timestamp

    evaluator = GroundTruthEvaluator(
        runs_dir=runs_path,
        ground_truth_dir=gt_path,
        output_dir=output_dir,
        enable_llm_judge=enable_llm_judge,
        max_workers=max_workers,
    )

    return evaluator.evaluate_all()
