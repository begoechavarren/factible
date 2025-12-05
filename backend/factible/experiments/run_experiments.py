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

app = typer.Typer(help="Run systematic experiments for Factible evaluation")
_logger = logging.getLogger(__name__)

# Default config file location
DEFAULT_CONFIG = Path("factible/experiments/experiments_config.yaml")


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
    video: dict, experiment: dict, settings: dict, dry_run: bool = False
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

    _logger.info("=" * 80)
    _logger.info(f"🚀 {full_name}")
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
        experiments = [e for e in experiments if e["name"] == experiment]
        if not experiments:
            _logger.error(f"Experiment '{experiment}' not found in config")
            raise typer.Exit(1)

    # Plan
    total = len(videos) * len(experiments)
    _logger.info(f"\n📊 Planning {total} runs:")
    _logger.info(f"   {len(videos)} videos × {len(experiments)} experiments\n")

    if dry_run:
        _logger.info("🔍 DRY RUN MODE\n")

    # Execute
    results = []
    for vid in videos:
        for exp in experiments:
            success = run_single(vid, exp, settings, dry_run)
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
