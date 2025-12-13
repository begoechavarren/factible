#!/usr/bin/env python
"""
Unified CLI for Factible experiments.

Provides commands for running experiments, analyzing results, and evaluating against ground truth.
"""

import logging
from typing import Annotated

import typer
from dotenv import load_dotenv

# Import command functions directly
from factible.experiments.run_experiments import run as run_experiments_command
from factible.experiments.run_evaluator import evaluate_runs

app = typer.Typer(
    help="Factible Experiments - Run and analyze fact-checking experiments",
    no_args_is_help=True,
)

# Add commands directly (not as sub-apps)
app.command(name="run", help="Run fact-checking experiments")(run_experiments_command)


@app.command(name="evaluate", help="Evaluate experiment runs against ground truth")
def evaluate_command(
    runs_dir: Annotated[
        str,
        typer.Option(
            help="Path to runs directory (e.g., 'factible/experiments/data/runs/20251213_114139_baseline')"
        ),
    ],
    ground_truth_dir: Annotated[
        str,
        typer.Option(help="Path to ground truth directory"),
    ] = "factible/experiments/data/ground_truth",
):
    """
    Evaluate experiment runs against manually annotated ground truth.

    The evaluator will:
    1. Find videos that exist in both runs_dir and ground_truth_dir
    2. Compare extracted claims vs ground truth claims
    3. Compare verdicts/stances vs ground truth verdicts
    4. Calculate metrics: precision, recall, F1, importance MAE, stance accuracy
    5. Save results to: data/eval_results/{run_name}/{timestamp}/

    Each evaluation creates a timestamped subdirectory, allowing you to track
    evaluation history over time.

    Examples:
        # Evaluate baseline runs
        factible-experiments evaluate \\
            --runs-dir factible/experiments/data/runs/20251213_114139_baseline

        # Evaluate with custom ground truth location
        factible-experiments evaluate \\
            --runs-dir factible/experiments/data/runs/my_experiment \\
            --ground-truth-dir factible/experiments/data/ground_truth
    """
    evaluate_runs(
        runs_dir=runs_dir,
        ground_truth_dir=ground_truth_dir,
    )


@app.callback()
def main(
    log_level: Annotated[str, typer.Option(help="Logging level")] = "INFO",
):
    """
    Factible Experiments - Run and analyze fact-checking experiments.

    Examples:
        # Run experiments
        factible-experiments run --experiment baseline

        # Analyze results
        factible-experiments analyze

        # Evaluate against ground truth
        factible-experiments evaluate --runs-dir factible/experiments/data/runs/20251213_114139_baseline

        # Analyze specific runs
        factible-experiments analyze --runs-dir my_results/
    """
    # Setup global logging
    logging.basicConfig(level=log_level, format="%(message)s")
    load_dotenv()


if __name__ == "__main__":
    app()
