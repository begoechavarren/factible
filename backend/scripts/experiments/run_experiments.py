#!/usr/bin/env python
"""
Run systematic experiments for Factible evaluation.

This script reads experiments_config.yaml and runs the fact-checking pipeline
on all defined videos with all specified experiment configurations.

Usage:
    python scripts/experiments/run_experiments.py
    python scripts/experiments/run_experiments.py --experiment baseline_gpt4o_mini
    python scripts/experiments/run_experiments.py --video example_video_1
    python scripts/experiments/run_experiments.py --dry-run
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Add backend directory to path to import factible
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from factible.models.config import (
    CLAIM_EXTRACTOR_MODEL,
    EVIDENCE_EXTRACTOR_MODEL,
    OUTPUT_GENERATOR_MODEL,
    QUERY_GENERATOR_MODEL,
)
from factible.run import run_factible

_logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict[str, Any]:
    """Load experiment configuration from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Validate required sections
    if "videos" not in config or not config["videos"]:
        raise ValueError("Config must contain at least one video in 'videos' section")
    if "experiments" not in config or not config["experiments"]:
        raise ValueError(
            "Config must contain at least one experiment in 'experiments' section"
        )

    return config


def should_run_experiment(
    video: dict[str, Any], experiment: dict[str, Any]
) -> tuple[bool, str]:
    """
    Determine if an experiment should run on a video based on filters.

    Returns:
        (should_run, reason)
    """
    video_filter = experiment.get("video_filter", [])

    # Empty filter means run on all videos
    if not video_filter:
        return True, "no filter"

    # Check if any video tag matches the filter
    video_tags = set(video.get("tags", []))
    filter_tags = set(video_filter)

    if video_tags & filter_tags:
        return True, f"matches tags: {video_tags & filter_tags}"

    return False, f"no matching tags (video: {video_tags}, filter: {filter_tags})"


def run_single_experiment(
    video: dict[str, Any],
    experiment: dict[str, Any],
    settings: dict[str, Any],
    dry_run: bool = False,
) -> bool:
    """
    Run a single experiment on a video.

    Returns:
        True if successful, False if failed
    """
    video_id = video["id"]
    video_url = video["url"]
    exp_name = experiment["name"]

    # Check if should run based on filters
    should_run, reason = should_run_experiment(video, experiment)
    if not should_run:
        _logger.info(
            f"⊘ Skipping [{exp_name}] on [{video_id}]: {reason}",
        )
        return True  # Not a failure, just skipped

    # Construct experiment name: {experiment_name}_{video_id}
    full_experiment_name = f"{exp_name}_{video_id}"

    _logger.info("=" * 80)
    _logger.info(f"🚀 Starting experiment: {full_experiment_name}")
    _logger.info(f"   Video: {video.get('description', video_id)}")
    _logger.info(f"   URL: {video_url}")
    _logger.info(f"   Config: {experiment.get('description', exp_name)}")
    _logger.info("=" * 80)

    if dry_run:
        _logger.info("   [DRY RUN] Would execute with parameters:")
        _logger.info(f"     max_claims: {experiment.get('max_claims')}")
        _logger.info(
            f"     max_queries_per_claim: {experiment.get('max_queries_per_claim')}"
        )
        _logger.info(
            f"     max_results_per_query: {experiment.get('max_results_per_query')}"
        )
        _logger.info(f"     enable_search: {experiment.get('enable_search')}")
        return True

    try:
        result = run_factible(
            video_url=video_url,
            experiment_name=full_experiment_name,
            max_claims=experiment.get("max_claims", 5),
            enable_search=experiment.get("enable_search", True),
            max_queries_per_claim=experiment.get("max_queries_per_claim", 2),
            max_results_per_query=experiment.get("max_results_per_query", 3),
            headless_search=settings.get("headless_search", True),
        )

        _logger.info("✅ Experiment completed successfully")
        _logger.info(
            f"   Processed {result.extracted_claims.total_count} claims with "
            f"{len(result.claim_reports)} reports"
        )
        return True

    except Exception as exc:
        _logger.error(f"❌ Experiment failed: {exc}", exc_info=True)
        return False


def main():
    """Main entry point for experiment runner."""
    parser = argparse.ArgumentParser(
        description="Run systematic experiments for Factible evaluation"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "experiments_config.yaml",
        help="Path to experiments config YAML file",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        help="Run only the specified experiment by name",
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Run only on the specified video by ID",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be executed without running experiments",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(message)s",
    )

    # Load environment variables
    load_dotenv()

    # Show current model configuration
    _logger.info("📋 Current model configuration:")
    _logger.info(f"   Claim Extractor: {CLAIM_EXTRACTOR_MODEL.name}")
    _logger.info(f"   Query Generator: {QUERY_GENERATOR_MODEL.name}")
    _logger.info(f"   Evidence Extractor: {EVIDENCE_EXTRACTOR_MODEL.name}")
    _logger.info(f"   Output Generator: {OUTPUT_GENERATOR_MODEL.name}")
    _logger.info("")

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as exc:
        _logger.error(f"Failed to load config: {exc}")
        return 1

    videos = config["videos"]
    experiments = config["experiments"]
    settings = config.get("settings", {})

    # Filter videos if specified
    if args.video:
        videos = [v for v in videos if v["id"] == args.video]
        if not videos:
            _logger.error(f"Video '{args.video}' not found in config")
            return 1

    # Filter experiments if specified
    if args.experiment:
        experiments = [e for e in experiments if e["name"] == args.experiment]
        if not experiments:
            _logger.error(f"Experiment '{args.experiment}' not found in config")
            return 1

    # Calculate total runs
    total_runs = len(videos) * len(experiments)
    _logger.info(f"📊 Planning {total_runs} experiment runs:")
    _logger.info(f"   {len(videos)} videos × {len(experiments)} experiments")
    _logger.info("")

    if args.dry_run:
        _logger.info("🔍 DRY RUN MODE - No experiments will be executed")
        _logger.info("")

    # Run all experiment combinations
    results = []
    for video in videos:
        for experiment in experiments:
            success = run_single_experiment(
                video, experiment, settings, dry_run=args.dry_run
            )
            results.append(
                {
                    "video": video["id"],
                    "experiment": experiment["name"],
                    "success": success,
                }
            )

    # Summary
    _logger.info("")
    _logger.info("=" * 80)
    _logger.info("📈 EXPERIMENT RUN SUMMARY")
    _logger.info("=" * 80)

    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful

    _logger.info(f"Total runs: {len(results)}")
    _logger.info(f"✅ Successful: {successful}")
    _logger.info(f"❌ Failed: {failed}")

    if failed > 0:
        _logger.info("")
        _logger.info("Failed experiments:")
        for r in results:
            if not r["success"]:
                _logger.info(f"  - {r['experiment']} on {r['video']}")

    _logger.info("")
    _logger.info("Results saved to: experiments/runs/")
    _logger.info("See EVALUATION.md for analysis instructions")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
