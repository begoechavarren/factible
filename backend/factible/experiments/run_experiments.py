#!/usr/bin/env python
"""Run systematic experiments for Factible evaluation."""

import logging
from pathlib import Path
from typing import Annotated, Any, Optional

import typer
import yaml
from dotenv import load_dotenv

from factible.models.config import (
    CLAIM_EXTRACTOR_MODEL,
    EVIDENCE_EXTRACTOR_MODEL,
    OUTPUT_GENERATOR_MODEL,
    QUERY_GENERATOR_MODEL,
)
from factible.run import run_factible

app = typer.Typer(
    help="Run systematic experiments for Factible evaluation",
    no_args_is_help=True,
)
_logger = logging.getLogger(__name__)

# Default config file location
DEFAULT_CONFIG = Path("factible/experiments/experiments_config.yaml")


def expand_experiments(experiments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Expand experiments with list-valued parameters into individual configs.

    Example:
        Input:  {"name": "vary_claims", "max_claims": [1, 3, 5], "max_queries_per_claim": 2}
        Output: [
            {"name": "vary_claims_claims1", "max_claims": 1, "max_queries_per_claim": 2},
            {"name": "vary_claims_claims3", "max_claims": 3, "max_queries_per_claim": 2},
            {"name": "vary_claims_claims5", "max_claims": 5, "max_queries_per_claim": 2},
        ]
    """
    expanded = []

    # Parameters that can be varied
    SWEEP_PARAMS = {
        "max_claims": "claims",
        "max_queries_per_claim": "queries",
        "max_results_per_query": "results",
        "model_config": "model",
    }

    for exp in experiments:
        # Check if any parameter is a list
        list_params = {}
        for param, short_name in SWEEP_PARAMS.items():
            if param in exp and isinstance(exp[param], list):
                list_params[param] = (short_name, exp[param])

        # If no list params, keep as-is
        if not list_params:
            expanded.append(exp)
            continue

        # If multiple list params, raise error (not supported for clarity)
        if len(list_params) > 1:
            param_names = ", ".join(list_params.keys())
            _logger.error(
                f"Experiment '{exp['name']}' has multiple list parameters: {param_names}. "
                "Only one parameter can be varied per experiment group for clarity."
            )
            raise typer.Exit(1)

        # Expand the single list parameter
        param_name, (short_name, values) = list(list_params.items())[0]
        base_name = exp["name"]

        for value in values:
            # Create new config with this value
            new_exp = exp.copy()
            new_exp[param_name] = value
            new_exp["name"] = f"{base_name}_{short_name}{value}"

            # Add metadata about the sweep
            if "description" not in new_exp:
                new_exp["description"] = f"Sweep {param_name}={value}"
            else:
                new_exp["description"] = (
                    f"{new_exp['description']} (sweep {param_name}={value})"
                )

            expanded.append(new_exp)
            _logger.debug(f"  Expanded: {base_name} → {new_exp['name']}")

    return expanded


def load_config(config_path: Path) -> dict[str, Any]:
    """Load and validate experiment configuration."""
    if not config_path.exists():
        _logger.error(f"Config file not found: {config_path}")
        raise typer.Exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if "videos" not in config or not config["videos"]:
        _logger.error("Config must contain at least one video")
        raise typer.Exit(1)

    if "experiments" not in config or not config["experiments"]:
        _logger.error("Config must contain at least one experiment")
        raise typer.Exit(1)

    # Expand experiments with list-valued parameters
    original_count = len(config["experiments"])
    config["experiments"] = expand_experiments(config["experiments"])
    expanded_count = len(config["experiments"])

    if expanded_count > original_count:
        _logger.info(
            f"🔄 Expanded {original_count} experiment groups → {expanded_count} individual experiments"
        )

    return config


def should_run(video: dict, experiment: dict) -> tuple[bool, str]:
    """Check if experiment should run on video based on filters."""
    video_filter = experiment.get("video_filter", [])
    if not video_filter:
        return True, "no filter"

    video_tags = set(video.get("tags", []))
    filter_tags = set(video_filter)

    if video_tags & filter_tags:
        return True, f"matches {video_tags & filter_tags}"

    return False, "no match"


def run_single(
    video: dict,
    experiment: dict,
    settings: dict,
    dry_run: bool = False,
    current: int = 1,
    total: int = 1,
    runs_subdir: Optional[str] = None,
) -> bool:
    """Run a single experiment on a video."""
    video_id = video["id"]
    exp_name = experiment["name"]
    full_name = f"{exp_name}_{video_id}"

    # Check filters
    should, reason = should_run(video, experiment)
    if not should:
        _logger.info(f"⊘ Skipping [{exp_name}] on [{video_id}]: {reason}")
        return True

    # Check if this experiment already exists in the runs directory
    if runs_subdir:
        runs_dir = Path("factible/experiments/runs") / runs_subdir
        if runs_dir.exists():
            # Look for existing runs matching this experiment+video combination
            # Run directories are named: YYYYMMDD_HHMMSS_end_to_end_EXPERIMENTNAME
            existing_runs = list(runs_dir.glob(f"*_end_to_end_{full_name}"))
            if existing_runs:
                _logger.info(f"⏭️  Skipping [{full_name}] - already exists:")
                for existing_run in existing_runs:
                    _logger.info(f"     {existing_run.name}")
                return True

    _logger.info("=" * 80)
    _logger.info(f"🚀 Experiment {current}/{total}: {full_name}")
    _logger.info(f"   Video: {video.get('description', video_id)}")
    _logger.info(f"   URL: {video['url']}")
    _logger.info("=" * 80)

    if dry_run:
        _logger.info("   [DRY RUN] Parameters:")
        _logger.info(f"     max_claims: {experiment.get('max_claims', 5)}")
        _logger.info(
            f"     max_queries_per_claim: {experiment.get('max_queries_per_claim', 2)}"
        )
        _logger.info(
            f"     max_results_per_query: {experiment.get('max_results_per_query', 3)}"
        )
        return True

    try:
        result = run_factible(
            video_url=video["url"],
            experiment_name=full_name,
            max_claims=experiment.get("max_claims", 5),
            max_queries_per_claim=experiment.get("max_queries_per_claim", 2),
            max_results_per_query=experiment.get("max_results_per_query", 3),
            headless_search=settings.get("headless_search", True),
            runs_subdir=runs_subdir,
        )

        _logger.info(
            f"✅ Completed - {result.extracted_claims.total_count} claims, "
            f"{len(result.claim_reports)} reports"
        )
        return True

    except Exception as exc:
        _logger.error(f"❌ Failed: {exc}")
        _logger.exception("Experiment failed")
        return False


@app.command()
def run(
    config: Annotated[
        Path,
        typer.Option(
            help="Path to experiments config YAML file",
            exists=True,
            dir_okay=False,
        ),
    ] = DEFAULT_CONFIG,
    experiment: Annotated[
        Optional[str], typer.Option(help="Run only this experiment by name")
    ] = None,
    video: Annotated[
        Optional[str], typer.Option(help="Run only on this video by ID")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Preview without executing")
    ] = False,
    runs_subdir: Annotated[
        Optional[str],
        typer.Option(
            help="Subdirectory within runs/ to organize this experiment session"
        ),
    ] = None,
    log_level: Annotated[str, typer.Option(help="Logging level")] = "INFO",
):
    """
    Run experiments on YouTube videos for fact-checking evaluation.

    Examples:
        # Run all experiments
        python run_experiments.py

        # Run specific experiment
        python run_experiments.py --experiment baseline_gpt4o_mini

        # Run on specific video
        python run_experiments.py --video example_video_1

        # Dry run (preview)
        python run_experiments.py --dry-run
    """
    # Setup logging
    logging.basicConfig(level=log_level, format="%(message)s")
    load_dotenv()

    # Show model config
    _logger.info("📋 Current model configuration:")
    _logger.info(f"   Claim Extractor: {CLAIM_EXTRACTOR_MODEL.name}")
    _logger.info(f"   Query Generator: {QUERY_GENERATOR_MODEL.name}")
    _logger.info(f"   Evidence Extractor: {EVIDENCE_EXTRACTOR_MODEL.name}")
    _logger.info(f"   Output Generator: {OUTPUT_GENERATOR_MODEL.name}")

    # Load config
    cfg = load_config(config)
    videos = cfg["videos"]
    experiments = cfg["experiments"]
    settings = cfg.get("settings", {})

    # Filter if requested
    if video:
        videos = [v for v in videos if v["id"] == video]
        if not videos:
            _logger.error(f"Video '{video}' not found in config")
            raise typer.Exit(1)

    if experiment:
        # Filter by exact match OR prefix match (for auto-expanded experiments)
        experiments = [
            e
            for e in experiments
            if e["name"] == experiment or e["name"].startswith(f"{experiment}_")
        ]
        if not experiments:
            _logger.error(f"Experiment '{experiment}' not found in config")
            raise typer.Exit(1)

    # Auto-generate runs_subdir if not provided
    if runs_subdir is None:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if len(experiments) == 1:
            # Single experiment: use experiment name
            runs_subdir = f"{experiments[0]['name']}_{timestamp}"
        else:
            # Multiple experiments: check if they share a common base name
            # (e.g., vary_claims_claims1, vary_claims_claims3 -> vary_claims)
            first_name = experiments[0]["name"]
            if "_" in first_name and all(
                e["name"].startswith(first_name.rsplit("_", 1)[0]) for e in experiments
            ):
                # Use common prefix
                base_name = first_name.rsplit("_", 1)[0]
                runs_subdir = f"{base_name}_{timestamp}"
            else:
                # Unrelated experiments: use batch name
                runs_subdir = f"batch_{timestamp}"

        _logger.info(f"📁 Auto-generated runs directory: runs/{runs_subdir}/\n")

    # Plan
    total = len(videos) * len(experiments)
    _logger.info(f"\n📊 Planning {total} runs:")
    _logger.info(f"   {len(videos)} videos × {len(experiments)} experiments\n")

    if dry_run:
        _logger.info("🔍 DRY RUN MODE\n")

    # Execute
    results = []
    current = 0
    for vid in videos:
        for exp in experiments:
            current += 1
            success = run_single(
                vid, exp, settings, dry_run, current, total, runs_subdir
            )
            results.append(
                {"video": vid["id"], "experiment": exp["name"], "success": success}
            )

    # Summary
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful

    _logger.info("\n" + "=" * 80)
    _logger.info("📈 SUMMARY")
    _logger.info("=" * 80)
    _logger.info(f"Total: {len(results)}")
    _logger.info(f"✅ Successful: {successful}")
    _logger.info(f"❌ Failed: {failed}")

    if failed > 0:
        _logger.info("\nFailed experiments:")
        for r in results:
            if not r["success"]:
                _logger.info(f"  - {r['experiment']} on {r['video']}")

    _logger.info("\nResults saved to: experiments/runs/")

    if failed > 0:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
