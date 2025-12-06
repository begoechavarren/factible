#!/usr/bin/env python
"""
Unified CLI for Factible experiments.

Provides commands for running experiments and analyzing results.
"""

import logging
from typing import Annotated

import typer
from dotenv import load_dotenv

# Import subcommand modules
from factible.experiments import run_experiments
from factible.experiments.analyze import analyze_results

app = typer.Typer(
    help="Factible Experiments - Run and analyze fact-checking experiments",
    no_args_is_help=True,
)

# Add subcommands
app.add_typer(run_experiments.app, name="run", help="Run fact-checking experiments")
app.command(
    name="analyze", help="Analyze experiment results and generate visualizations"
)(analyze_results)


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

        # Analyze specific runs
        factible-experiments analyze --runs-dir my_results/
    """
    # Setup global logging
    logging.basicConfig(level=log_level, format="%(message)s")
    load_dotenv()


if __name__ == "__main__":
    app()
