from typing import Annotated

import typer
import logging
from dotenv import load_dotenv

from experiments.run_experiments import run as run_experiments_command
from experiments.evaluator.run_evaluator import evaluate_runs

app = typer.Typer(
    help="Factible Experiments - Run and analyze fact-checking experiments",
    no_args_is_help=True,
)

app.command(name="run", help="Run fact-checking experiments")(run_experiments_command)


@app.command(name="evaluate", help="Evaluate experiment runs against ground truth")
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
):
    """
    Evaluate experiment runs against manually annotated ground truth.

    Examples:
        # Enable LLM-as-judge evaluation (costs API calls)
        factible-experiments evaluate \\
            --runs-dir experiments/data/runs/my_experiment \\
            --enable-llm-judge \\
            --max-workers 2
    """
    evaluate_runs(
        runs_dir=runs_dir,
        ground_truth_dir=ground_truth_dir,
        enable_llm_judge=enable_llm_judge,
        max_workers=max_workers,
    )


@app.callback()
def main(
    log_level: Annotated[str, typer.Option(help="Logging level")] = "INFO",
):
    """
    Run and evaluate experiments.

    Examples:
        # Run experiments
        factible-experiments run --experiment baseline

        # Evaluate against ground truth
        factible-experiments evaluate --runs-dir experiments/data/runs/20251213_114139_baseline
    """
    # Setup global logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_dotenv()


if __name__ == "__main__":
    app()
