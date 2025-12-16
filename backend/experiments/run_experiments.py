import asyncio
import json
import logging
from pathlib import Path
from typing import Annotated, Any, Optional
from urllib.parse import parse_qs, urlparse

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

DEFAULT_CONFIG = Path("experiments/experiments_config.yaml")


def _youtube_video_id(url: str | None) -> str | None:
    """Extract the YouTube video ID from a URL."""
    if not url:
        return None
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        return parse_qs(parsed.query).get("v", [None])[0]
    if host.endswith("youtu.be"):
        return parsed.path.strip("/") or None
    return None


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

        # If multiple list params, raise error
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
            _logger.debug(f"  Expanded: {base_name} -> {new_exp['name']}")

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
            f"Expanded {original_count} experiment groups -> {expanded_count} individual experiments"
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


async def run_single(
    video: dict,
    experiment: dict,
    settings: dict,
    dry_run: bool = False,
    current: int = 1,
    total: int = 1,
    runs_subdir: Optional[str] = None,
) -> bool:
    """Run a single experiment on a video (async)."""
    video_id = video["id"]
    exp_name = experiment["name"]
    full_name = f"{exp_name}_{video_id}"
    youtube_id = _youtube_video_id(video.get("url"))

    # Check filters
    should, reason = should_run(video, experiment)
    if not should:
        _logger.info(f"Skipping [{exp_name}] on [{video_id}]: {reason}")
        return True

    # Check if this experiment already exists in the runs directory
    if runs_subdir:
        runs_dir = Path("experiments/data/runs") / runs_subdir
        if runs_dir.exists():
            for run_dir in runs_dir.iterdir():
                config_path = run_dir / "config.json"
                if not config_path.exists():
                    continue
                try:
                    with config_path.open() as cfg_file:
                        cfg = json.load(cfg_file)
                except Exception:  # noqa: BLE001
                    continue
                if cfg.get("experiment_name") == full_name:
                    _logger.info(f"Skipping [{full_name}] - already exists:")
                    _logger.info(f"     {run_dir.name}")
                    return True

    _logger.info("=" * 80)
    _logger.info(f"Experiment {current}/{total}: {full_name}")
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
        result = await run_factible(
            video_url=video["url"],
            experiment_name=full_name,
            max_claims=experiment.get("max_claims", 5),
            max_queries_per_claim=experiment.get("max_queries_per_claim", 2),
            max_results_per_query=experiment.get("max_results_per_query", 3),
            headless_search=settings.get("headless_search", True),
            runs_subdir=runs_subdir,
            run_label=youtube_id if youtube_id else None,
        )

        _logger.info(
            f"Completed - {result.extracted_claims.total_count} claims, "
            f"{len(result.claim_reports)} reports"
        )
        return True

    except Exception as exc:
        _logger.error(f"Failed: {exc}")
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
        factible-experiments run --experiment baseline --runs-subdir baseline_20251206_191228
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_dotenv()

    # Show model config
    _logger.info("Current model configuration:")
    _logger.info(f"   Claim Extractor: {CLAIM_EXTRACTOR_MODEL.name}")
    _logger.info(f"   Query Generator: {QUERY_GENERATOR_MODEL.name}")
    _logger.info(f"   Evidence Extractor: {EVIDENCE_EXTRACTOR_MODEL.name}")
    _logger.info(f"   Output Generator: {OUTPUT_GENERATOR_MODEL.name}")

    # Load config
    cfg = load_config(config)
    videos = cfg["videos"]
    experiments = cfg["experiments"]
    settings = cfg.get("settings", {})

    # Filter
    if video:
        videos = [v for v in videos if v["id"] == video]
        if not videos:
            _logger.error(f"Video '{video}' not found in config")
            raise typer.Exit(1)

    if experiment:
        # Filter by exact match or prefix match (for auto-expanded experiments)
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
            runs_subdir = f"{timestamp}_{experiments[0]['name']}"
        else:
            # Multiple experiments: check if they share a common base name
            first_name = experiments[0]["name"]
            if "_" in first_name and all(
                e["name"].startswith(first_name.rsplit("_", 1)[0]) for e in experiments
            ):
                # Use common prefix
                base_name = first_name.rsplit("_", 1)[0]
                runs_subdir = f"{timestamp}_{base_name}"
            else:
                # Unrelated experiments: use batch name
                runs_subdir = f"{timestamp}_batch"

        _logger.info(f"Auto-generated runs directory: runs/{runs_subdir}/\n")

    # Plan: calculate actual runs considering video_filter
    run_plan: list[tuple[dict[str, Any], dict[str, Any]]] = []
    filtered_videos = set()
    for vid in videos:
        for exp in experiments:
            should, reason = should_run(vid, exp)
            if should:
                run_plan.append((vid, exp))
                filtered_videos.add(vid["id"])
            else:
                _logger.info(f"Skipping [{exp['name']}] on [{vid['id']}]: {reason}")

    actual_runs = len(run_plan)

    total = actual_runs
    _logger.info(f"\nPlanning {total} runs:")
    _logger.info(
        f"   {len(filtered_videos)} videos × {len(experiments)} experiments (after applying video_filter)\n"
    )

    if dry_run:
        _logger.info("DRY RUN MODE\n")

    # Execute
    async def run_all_experiments():
        """Run all experiments asynchronously."""
        results = []
        for idx, (vid, exp) in enumerate(run_plan, start=1):
            success = await run_single(
                vid, exp, settings, dry_run, idx, total, runs_subdir
            )
            results.append(
                {"video": vid["id"], "experiment": exp["name"], "success": success}
            )
        return results

    results = asyncio.run(run_all_experiments())

    # Summary
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful

    _logger.info("\n" + "=" * 80)
    _logger.info("SUMMARY")
    _logger.info("=" * 80)
    _logger.info(f"Total: {len(results)}")
    _logger.info(f"Successful: {successful}")
    _logger.info(f"Failed: {failed}")

    if failed > 0:
        _logger.info("\nFailed experiments:")
        for r in results:
            if not r["success"]:
                _logger.info(f"  - {r['experiment']} on {r['video']}")

    _logger.info("\nResults saved to: experiments/data/runs/")

    if failed > 0:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
