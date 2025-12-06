#!/usr/bin/env python
"""
Analyze experiment results with filtering and visualization.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

import pandas as pd
import typer
import matplotlib.pyplot as plt
import seaborn as sns

_logger = logging.getLogger(__name__)

# Set style for all plots
sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10

# Default paths
DEFAULT_RUNS_DIR = Path("factible/experiments/runs")
DEFAULT_OUTPUT_BASE = Path("factible/experiments/analysis")


def load_all_runs(runs_dir: Path) -> pd.DataFrame:
    """Load all experiment runs from the specified directory."""
    runs = []

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        # Load config and metrics
        config_path = run_dir / "config.json"
        metrics_path = run_dir / "metrics.json"

        if not all([config_path.exists(), metrics_path.exists()]):
            continue

        with open(config_path) as f:
            config = json.load(f)
        with open(metrics_path) as f:
            metrics = json.load(f)

        # Load optional files
        outputs = {}
        outputs_path = run_dir / "outputs.json"

        if outputs_path.exists():
            with open(outputs_path) as f:
                outputs = json.load(f)

        # Extract key metrics
        experiment_name = config.get("experiment_name", "unknown")
        video_url = config.get("video_url", "")

        run_data = {
            "run_id": config.get("run_id", run_dir.name),
            "run_dir": str(run_dir),
            "experiment": experiment_name,
            "video_url": video_url,
            "video_id": video_url.split("=")[-1] if video_url else "",
            # Config params
            "max_claims": config.get("max_claims", 0),
            "max_queries_per_claim": config.get("max_queries_per_claim", 0),
            "max_results_per_query": config.get("max_results_per_query", 0),
            # Timing
            "total_time_seconds": metrics["timing"]["total_seconds"],
            "transcript_time": metrics["timing"].get(
                "Step 1: Transcript extraction", 0
            ),
            "claim_extraction_time": metrics["timing"].get(
                "Step 2: Claim extraction", 0
            ),
            "search_time": metrics["timing"].get(
                "Step 3: Query generation + search for all claims", 0
            ),
            "output_generation_time": metrics["timing"].get(
                "Step 4: Output generation", 0
            ),
            # LLM metrics
            "llm_total_calls": metrics.get("llm", {}).get("total_calls", 0),
            "llm_total_cost": metrics.get("llm", {}).get("total_cost_usd", 0),
            "llm_total_latency": metrics.get("llm", {}).get("total_latency_seconds", 0),
            # Results
            "claims_extracted": metrics.get("claims_extracted", 0),
            "success": metrics.get("success", False),
            # Claims analysis (if outputs available)
            "claims_processed": (
                len(outputs.get("final_output", {}).get("claim_reports", []))
                if outputs
                else 0
            ),
        }

        # Add claim-level stance distribution if available
        if outputs and "final_output" in outputs:
            claim_reports = outputs["final_output"].get("claim_reports", [])
            stances = [
                c.get("overall_stance")
                for c in claim_reports
                if c.get("overall_stance")
            ]
            run_data["supports_count"] = stances.count("supports")
            run_data["refutes_count"] = stances.count("refutes")
            run_data["mixed_count"] = stances.count("mixed")
            run_data["unclear_count"] = stances.count("unclear")

            # Confidence distribution
            confidences = [
                c.get("verdict_confidence")
                for c in claim_reports
                if c.get("verdict_confidence")
            ]
            run_data["high_confidence_count"] = confidences.count("high")
            run_data["medium_confidence_count"] = confidences.count("medium")
            run_data["low_confidence_count"] = confidences.count("low")

            # Source statistics
            sources = [c.get("total_sources", 0) for c in claim_reports]
            run_data["avg_sources_per_claim"] = (
                sum(sources) / len(sources) if sources else 0
            )
            run_data["total_sources"] = sum(sources)

        runs.append(run_data)

    return pd.DataFrame(runs)


def load_llm_calls_detail(
    runs_dir: Path, run_ids: Optional[list[str]] = None
) -> pd.DataFrame:
    """Load detailed LLM call data from filtered runs."""
    all_calls = []

    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue

        # Filter by run IDs if provided
        if run_ids and run_dir.name not in run_ids:
            continue

        llm_calls_path = run_dir / "llm_calls.json"
        config_path = run_dir / "config.json"

        if not (llm_calls_path.exists() and config_path.exists()):
            continue

        with open(config_path) as f:
            config = json.load(f)
        with open(llm_calls_path) as f:
            llm_calls = json.load(f)

        for call in llm_calls:
            call_data = {
                "run_id": config.get("run_id", run_dir.name),
                "experiment": config.get("experiment_name", "unknown"),
                "component": call.get("component"),
                "model": call.get("model"),
                "latency_seconds": call.get("latency_seconds", 0),
                "input_tokens": call.get("input_tokens_estimated", 0),
                "output_tokens": call.get("output_tokens_estimated", 0),
                "total_tokens": call.get("input_tokens_estimated", 0)
                + call.get("output_tokens_estimated", 0),
                "cost_usd": call.get("cost_usd", 0),
            }
            all_calls.append(call_data)

    return pd.DataFrame(all_calls)


def visualize_overview(df: pd.DataFrame, output_dir: Path):
    """Generate overview visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        f"Experiment Results Overview ({len(df)} runs)", fontsize=16, fontweight="bold"
    )

    # 1. Total time per experiment
    ax = axes[0, 0]
    df_sorted = df.sort_values("total_time_seconds")
    df_sorted.plot(
        x="experiment",
        y="total_time_seconds",
        kind="barh",
        ax=ax,
        legend=False,
        color="steelblue",
    )
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Experiment")
    ax.set_title("Total Execution Time per Experiment")

    # 2. Time breakdown by component
    ax = axes[0, 1]
    time_cols = [
        "transcript_time",
        "claim_extraction_time",
        "search_time",
        "output_generation_time",
    ]
    df[time_cols].mean().plot(kind="bar", ax=ax, color="coral")
    ax.set_xlabel("Component")
    ax.set_ylabel("Average Time (seconds)")
    ax.set_title("Average Time by Pipeline Component")
    ax.set_xticklabels(
        ["Transcript", "Claim Extraction", "Search", "Output Generation"], rotation=45
    )

    # 3. LLM calls per experiment
    ax = axes[1, 0]
    df.plot(
        x="experiment",
        y="llm_total_calls",
        kind="barh",
        ax=ax,
        legend=False,
        color="mediumseagreen",
    )
    ax.set_xlabel("LLM Calls")
    ax.set_ylabel("Experiment")
    ax.set_title("Total LLM Calls per Experiment")

    # 4. Cost/tokens per experiment
    ax = axes[1, 1]
    if df["llm_total_cost"].sum() > 0:
        df.plot(
            x="experiment",
            y="llm_total_cost",
            kind="barh",
            ax=ax,
            legend=False,
            color="gold",
        )
        ax.set_xlabel("Cost (USD)")
    else:
        df["llm_tokens_approx"] = df["llm_total_calls"] * 1000
        df.plot(
            x="experiment",
            y="llm_tokens_approx",
            kind="barh",
            ax=ax,
            legend=False,
            color="gold",
        )
        ax.set_xlabel("Approx. Tokens (thousands)")
    ax.set_ylabel("Experiment")
    ax.set_title("LLM Cost per Experiment")

    plt.tight_layout()
    plt.savefig(output_dir / "01_overview.png", dpi=300, bbox_inches="tight")
    _logger.info(f"✅ Saved: {output_dir / '01_overview.png'}")
    plt.close()


def visualize_component_breakdown(llm_df: pd.DataFrame, output_dir: Path):
    """Visualize LLM performance by component."""
    if llm_df.empty:
        _logger.warning("⚠️  No LLM call data available")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("LLM Performance by Component", fontsize=16, fontweight="bold")

    # 1. Latency by component
    ax = axes[0]
    component_latency = llm_df.groupby("component")["latency_seconds"].agg(
        ["mean", "std"]
    )
    component_latency["mean"].plot(
        kind="bar", ax=ax, yerr=component_latency["std"], capsize=5, color="skyblue"
    )
    ax.set_ylabel("Latency (seconds)")
    ax.set_xlabel("Component")
    ax.set_title("Average LLM Latency per Component")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    # 2. Token usage by component
    ax = axes[1]
    component_tokens = llm_df.groupby("component")[
        ["input_tokens", "output_tokens"]
    ].sum()
    component_tokens.plot(
        kind="bar", stacked=True, ax=ax, color=["lightcoral", "lightgreen"]
    )
    ax.set_ylabel("Tokens")
    ax.set_xlabel("Component")
    ax.set_title("Token Usage by Component")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(["Input Tokens", "Output Tokens"])

    # 3. Call count by component
    ax = axes[2]
    component_calls = llm_df.groupby("component").size()
    component_calls.plot(kind="bar", ax=ax, color="mediumorchid")
    ax.set_ylabel("Number of Calls")
    ax.set_xlabel("Component")
    ax.set_title("LLM Call Count by Component")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_dir / "02_component_breakdown.png", dpi=300, bbox_inches="tight")
    _logger.info(f"✅ Saved: {output_dir / '02_component_breakdown.png'}")
    plt.close()


def visualize_claims_analysis(df: pd.DataFrame, output_dir: Path):
    """Visualize claim-level analysis."""
    if "supports_count" not in df.columns:
        _logger.warning("⚠️  No claim stance data available")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Claim Analysis", fontsize=16, fontweight="bold")

    # 1. Stance distribution
    ax = axes[0]
    stance_cols = ["supports_count", "refutes_count", "mixed_count", "unclear_count"]
    stance_totals = df[stance_cols].sum()
    stance_totals.plot(kind="pie", ax=ax, autopct="%1.1f%%", startangle=90)
    ax.set_ylabel("")
    ax.set_title("Overall Stance Distribution")
    ax.legend(["Supports", "Refutes", "Mixed", "Unclear"], loc="best")

    # 2. Confidence distribution
    ax = axes[1]
    if "high_confidence_count" in df.columns:
        conf_cols = [
            "high_confidence_count",
            "medium_confidence_count",
            "low_confidence_count",
        ]
        conf_totals = df[conf_cols].sum()
        conf_totals.plot(kind="bar", ax=ax, color=["darkgreen", "orange", "crimson"])
        ax.set_ylabel("Count")
        ax.set_xlabel("Confidence Level")
        ax.set_title("Verdict Confidence Distribution")
        ax.set_xticklabels(["High", "Medium", "Low"], rotation=0)

    # 3. Sources per claim
    ax = axes[2]
    if "avg_sources_per_claim" in df.columns:
        df.plot(
            x="experiment",
            y="avg_sources_per_claim",
            kind="barh",
            ax=ax,
            legend=False,
            color="teal",
        )
        ax.set_xlabel("Average Sources per Claim")
        ax.set_ylabel("Experiment")
        ax.set_title("Evidence Coverage")

    plt.tight_layout()
    plt.savefig(output_dir / "03_claims_analysis.png", dpi=300, bbox_inches="tight")
    _logger.info(f"✅ Saved: {output_dir / '03_claims_analysis.png'}")
    plt.close()


def generate_summary_stats(df: pd.DataFrame, llm_df: pd.DataFrame, output_dir: Path):
    """Generate summary statistics table."""
    summary = {
        "Total Experiments": len(df),
        "Successful Runs": df["success"].sum(),
        "Failed Runs": (~df["success"]).sum(),
        "Total Claims Extracted": df["claims_extracted"].sum(),
        "Total Claims Processed": df["claims_processed"].sum(),
        "Total LLM Calls": df["llm_total_calls"].sum(),
        "Total Time (minutes)": df["total_time_seconds"].sum() / 60,
        "Avg Time per Experiment (seconds)": df["total_time_seconds"].mean(),
        "Total Cost (USD)": df["llm_total_cost"].sum(),
        "Avg Cost per Experiment (USD)": df["llm_total_cost"].mean(),
    }

    # Add component-specific stats
    if not llm_df.empty:
        for component in [
            "claim_extraction",
            "query_generation",
            "evidence_extraction",
            "verdict_generation",
        ]:
            component_data = llm_df[llm_df["component"] == component]
            if not component_data.empty:
                summary[f"Avg Latency - {component.replace('_', ' ').title()} (s)"] = (
                    component_data["latency_seconds"].mean()
                )

    # Save to file
    with open(output_dir / "summary_statistics.txt", "w") as f:
        f.write("=" * 60 + "\n")
        f.write("EXPERIMENT RESULTS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        for key, value in summary.items():
            if isinstance(value, float):
                f.write(f"{key:.<50} {value:.2f}\n")
            else:
                f.write(f"{key:.<50} {value}\n")

    _logger.info(f"✅ Saved: {output_dir / 'summary_statistics.txt'}")
    return summary


def analyze_results(
    runs_dir: Annotated[
        Path,
        typer.Option(
            help="Directory containing experiment runs to analyze",
            exists=True,
            dir_okay=True,
            file_okay=False,
        ),
    ] = DEFAULT_RUNS_DIR,
    name: Annotated[
        Optional[str],
        typer.Option(
            help="Custom name for this analysis (creates timestamped subdirectory)"
        ),
    ] = None,
):
    """
    Analyze experiment results and generate visualizations.

    Creates a timestamped subdirectory in factible/experiments/analysis/
    to preserve each analysis run.

    Examples:
        # Analyze all runs in default directory
        factible-experiments analyze

        # Analyze runs from custom directory
        factible-experiments analyze --runs-dir factible/experiments/runs/baseline_only

        # Analyze with custom name
        factible-experiments analyze --name baseline_results

        # Analyze custom directory with custom name
        factible-experiments analyze --runs-dir runs/december --name december_analysis
    """
    # Generate timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if name:
        dir_name = f"{timestamp}_{name}"
    else:
        dir_name = f"{timestamp}_analysis"

    output_dir = DEFAULT_OUTPUT_BASE / dir_name
    _logger.info(f"📁 Output directory: {output_dir}")
    _logger.info(f"📂 Analyzing runs from: {runs_dir}\n")

    # Create output directory
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load data
    _logger.info("📊 Loading experiment data...")
    df = load_all_runs(runs_dir)

    if df.empty:
        _logger.error(f"❌ No experiment data found in: {runs_dir}")
        raise typer.Exit(1)

    # Load LLM call details for filtered runs
    run_ids = df["run_id"].tolist()
    llm_df = load_llm_calls_detail(runs_dir, run_ids)

    _logger.info(f"✅ Loaded {len(df)} experiment runs")
    _logger.info(f"✅ Loaded {len(llm_df)} LLM calls\n")

    # Generate visualizations
    _logger.info("📈 Generating visualizations...\n")
    visualize_overview(df, output_dir)
    visualize_component_breakdown(llm_df, output_dir)
    visualize_claims_analysis(df, output_dir)

    # Generate summary statistics
    _logger.info("\n📋 Generating summary statistics...")
    generate_summary_stats(df, llm_df, output_dir)

    # Save raw data
    df.to_csv(output_dir / "experiments_data.csv", index=False)
    if not llm_df.empty:
        llm_df.to_csv(output_dir / "llm_calls_data.csv", index=False)
    _logger.info("✅ Saved raw data CSVs\n")

    _logger.info("=" * 60)
    _logger.info(f"✨ Analysis complete! Check the '{output_dir}' directory")
    _logger.info("=" * 60)
    _logger.info("\nAnalyzed experiments:")
    for exp in df["experiment"].unique():
        count = len(df[df["experiment"] == exp])
        _logger.info(f"  - {exp}: {count} run(s)")


# This module is imported by cli.py to add the analyze command
