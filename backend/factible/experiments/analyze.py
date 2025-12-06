#!/usr/bin/env python
"""
Analyze experiment results with filtering and visualization.
"""

import json
import logging
import re
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

import pandas as pd
import typer
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import yaml

from factible.models.config import (
    CLAIM_EXTRACTOR_MODEL,
    EVIDENCE_EXTRACTOR_MODEL,
    OUTPUT_GENERATOR_MODEL,
    QUERY_GENERATOR_MODEL,
)
from factible.models.llm import ModelChoice
from factible.experiments.phase1_visualizations import (
    enrich_experiment_data,
    plot_ofat_analysis,
    plot_strategy_comparison,
    plot_scalability_analysis,
    plot_cost_breakdown,
    plot_evidence_quality_analysis,
)

_logger = logging.getLogger(__name__)

# Set style for all plots
sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10

# Default paths
DEFAULT_RUNS_DIR = Path("factible/experiments/runs")
DEFAULT_OUTPUT_BASE = Path("factible/experiments/analysis")
DEFAULT_CONFIG_PATH = Path("factible/experiments/experiments_config.yaml")


def _model_display(choice_name: Optional[str]) -> str:
    if not choice_name:
        return "unknown"
    try:
        model_choice = ModelChoice[choice_name]
        return model_choice.value.model_name
    except KeyError:
        return choice_name.replace("_", " ").lower()


DEFAULT_MODEL_CHOICES = {
    "claim_extractor": CLAIM_EXTRACTOR_MODEL.name,
    "query_generator": QUERY_GENERATOR_MODEL.name,
    "evidence_extractor": EVIDENCE_EXTRACTOR_MODEL.name,
    "output_generator": OUTPUT_GENERATOR_MODEL.name,
}


def _parse_duration_to_seconds(duration_str: Optional[str]) -> Optional[float]:
    if not duration_str:
        return None
    minutes = 0
    seconds = 0
    match_minutes = re.search(r"(\d+)m", duration_str)
    match_seconds = re.search(r"(\d+)s", duration_str)
    if match_minutes:
        minutes = int(match_minutes.group(1))
    if match_seconds:
        seconds = int(match_seconds.group(1))
    total_seconds = minutes * 60 + seconds
    return float(total_seconds) if total_seconds > 0 else None


def _load_video_metadata(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, dict]:
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        data = yaml.safe_load(f) or {}
    metadata: dict[str, dict] = {}
    for video in data.get("videos", []):
        entry = video.copy()
        entry["duration_seconds"] = _parse_duration_to_seconds(video.get("duration"))
        video_id = video.get("id")
        video_url = video.get("url")
        if video_url:
            metadata[video_url] = entry
        if video_id:
            metadata[video_id] = entry
    return metadata


VIDEO_METADATA = _load_video_metadata()


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
        raw_model_config = config.get("model_config", {}) or {}
        model_config = DEFAULT_MODEL_CHOICES.copy()
        model_config.update(raw_model_config)
        model_summary = config.get("model_summary")
        if not model_summary and model_config:
            component_aliases = {
                "claim_extractor": "CE",
                "query_generator": "QG",
                "evidence_extractor": "EE",
                "output_generator": "OG",
            }
            parts = []
            for key, alias in component_aliases.items():
                choice_name = model_config.get(key)
                if choice_name:
                    display = _model_display(choice_name)
                    parts.append(f"{alias}:{display}")
            if parts:
                model_summary = " | ".join(parts)

        if not model_summary:
            model_summary = " | ".join(
                [
                    f"CE:{CLAIM_EXTRACTOR_MODEL.value.model_name}",
                    f"QG:{QUERY_GENERATOR_MODEL.value.model_name}",
                    f"EE:{EVIDENCE_EXTRACTOR_MODEL.value.model_name}",
                    f"OG:{OUTPUT_GENERATOR_MODEL.value.model_name}",
                ]
            )

        timing = metrics.get("timing", {})

        def _sum_matching(keyword: str) -> float:
            return sum(
                float(duration)
                for label, duration in timing.items()
                if label != "total_seconds" and keyword in label
            )

        query_generation_time = _sum_matching("Query generation for claim")
        google_search_time = _sum_matching("Google Search API")
        reliability_time = _sum_matching("Reliability assessment")
        content_fetch_time = _sum_matching("Content fetching")
        evidence_extraction_time = _sum_matching("Evidence extraction")

        video_metadata = VIDEO_METADATA.get(video_url) or VIDEO_METADATA.get(
            video_url.split("=")[-1]
        )
        transcript_tokens = config.get("transcript_tokens")
        if transcript_tokens is None:
            transcript_length = config.get("transcript_length")
            if transcript_length:
                transcript_tokens = int(max(1, transcript_length / 4))

        video_duration_seconds = config.get("video_duration_seconds")
        if video_duration_seconds is None and video_metadata:
            video_duration_seconds = video_metadata.get("duration_seconds")

        run_data = {
            "run_id": config.get("run_id", run_dir.name),
            "run_dir": str(run_dir),
            "experiment": experiment_name,
            "video_url": video_url,
            "video_id": video_url.split("=")[-1] if video_url else "",
            "video_title": config.get("video_title")
            or (video_metadata.get("description") if video_metadata else None),
            "model_config": model_config,
            "model_summary": model_summary,
            # Config params
            "max_claims": config.get("max_claims", 0),
            "max_queries_per_claim": config.get("max_queries_per_claim", 0),
            "max_results_per_query": config.get("max_results_per_query", 0),
            "headless_search": config.get("headless_search"),
            "component": config.get("component"),
            "transcript_length": config.get("transcript_length"),
            "transcript_tokens": transcript_tokens,
            "video_duration_seconds": video_duration_seconds,
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
            "query_generation_time": query_generation_time,
            "google_search_time": google_search_time,
            "reliability_time": reliability_time,
            "content_fetch_time": content_fetch_time,
            "evidence_extraction_time": evidence_extraction_time,
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
    """Generate overview visualizations with run parameters."""
    ordered_df = df.sort_values("total_time_seconds").reset_index(drop=True)

    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 2, height_ratios=[1, 1, 1], figure=fig)
    fig.suptitle(
        f"Experiment Results Overview ({len(df)} runs)", fontsize=16, fontweight="bold"
    )

    # 1. Total time per experiment
    ax = fig.add_subplot(gs[0, 0])
    ordered_df.plot(
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
    ax = fig.add_subplot(gs[0, 1])
    component_columns = [
        ("Transcript", "transcript_time"),
        ("Claim Extraction", "claim_extraction_time"),
        ("Query Generation", "query_generation_time"),
        ("Google Search", "google_search_time"),
        ("Reliability", "reliability_time"),
        ("Content Fetch", "content_fetch_time"),
        ("Evidence Extraction", "evidence_extraction_time"),
        ("Output Generation", "output_generation_time"),
    ]
    component_labels = []
    component_values = []
    for label, column in component_columns:
        if column in ordered_df.columns:
            component_labels.append(label)
            component_values.append(ordered_df[column].fillna(0).mean())
    ax.bar(component_labels, component_values, color="coral")
    ax.set_xlabel("Component")
    ax.set_ylabel("Average Time (seconds)")
    ax.set_title("Average Time by Pipeline Component")
    ax.set_xticks(range(len(component_labels)))
    ax.set_xticklabels(component_labels, rotation=45, ha="right")

    # 3. LLM calls per experiment
    ax = fig.add_subplot(gs[1, 0])
    ordered_df.plot(
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
    ax = fig.add_subplot(gs[1, 1])
    if ordered_df["llm_total_cost"].sum() > 0:
        ordered_df.plot(
            x="experiment",
            y="llm_total_cost",
            kind="barh",
            ax=ax,
            legend=False,
            color="gold",
        )
        ax.set_xlabel("Cost (USD)")
        ax.set_title("LLM Cost per Experiment")
    else:
        ordered_df["llm_tokens_approx"] = ordered_df["llm_total_calls"] * 1000
        ordered_df.plot(
            x="experiment",
            y="llm_tokens_approx",
            kind="barh",
            ax=ax,
            legend=False,
            color="gold",
        )
        ax.set_xlabel("Approx. Tokens (thousands)")
        ax.set_title("LLM Tokens per Experiment")
    ax.set_ylabel("Experiment")

    # 5. Parameter table for each run
    table_ax = fig.add_subplot(gs[2, :])
    table_ax.axis("off")
    table_ax.set_title("Run Configuration", loc="left", fontweight="bold")

    video_label = (
        ordered_df["video_title"]
        if "video_title" in ordered_df.columns
        else (
            ordered_df["video_id"]
            if "video_id" in ordered_df.columns and ordered_df["video_id"].any()
            else ordered_df["video_url"]
        )
    )

    model_configs_series = ordered_df.get("model_config")
    if model_configs_series is None:
        model_configs_series = pd.Series([DEFAULT_MODEL_CHOICES] * len(ordered_df))
    ce_models = model_configs_series.apply(
        lambda cfg: _model_display((cfg or {}).get("claim_extractor"))
    )
    qg_models = model_configs_series.apply(
        lambda cfg: _model_display((cfg or {}).get("query_generator"))
    )
    ee_models = model_configs_series.apply(
        lambda cfg: _model_display((cfg or {}).get("evidence_extractor"))
    )
    og_models = model_configs_series.apply(
        lambda cfg: _model_display((cfg or {}).get("output_generator"))
    )

    def _format_duration(seconds: Optional[float]) -> str:
        if seconds is None or pd.isna(seconds):
            return "?"
        total_seconds = float(seconds)
        minutes = int(total_seconds // 60)
        secs = int(total_seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def _wrap_text(value: str, width: int) -> str:
        if value is None:
            return ""
        wrapped = textwrap.wrap(str(value), width=width)
        return "\n".join(wrapped) if wrapped else str(value)

    video_seconds_series = ordered_df.get(
        "video_duration_seconds", pd.Series([None] * len(ordered_df))
    )
    table_df = pd.DataFrame(
        {
            "Experiment": ordered_df["experiment"],
            "Video": video_label,
            "Video ID": ordered_df["video_id"],
            "CE Model": ce_models,
            "QG Model": qg_models,
            "EE Model": ee_models,
            "OG Model": og_models,
            "Total Time (s)": ordered_df["total_time_seconds"],
            "Total Time (mm:ss)": ordered_df["total_time_seconds"].apply(
                _format_duration
            ),
            "Max Claims": ordered_df["max_claims"],
            "Queries/Claim": ordered_df["max_queries_per_claim"],
            "Results/Query": ordered_df["max_results_per_query"],
            "Transcript Tokens": ordered_df.get(
                "transcript_tokens", pd.Series([None] * len(ordered_df))
            ),
            "Video Length (mm:ss)": video_seconds_series.apply(_format_duration),
        }
    )

    # Format text-heavy columns for readability
    table_df["Experiment"] = (
        table_df["Experiment"]
        .astype(str)
        .apply(lambda text: _wrap_text(text, width=25))
    )
    table_df["Video"] = (
        table_df["Video"].astype(str).apply(lambda text: _wrap_text(text, width=32))
    )
    table_df["Video ID"] = table_df["Video ID"].astype(str)
    if "Transcript Tokens" in table_df:
        table_df["Transcript Tokens"] = (
            table_df["Transcript Tokens"].astype(float).round().astype("Int64")
        )

    table = table_ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center",
        colWidths=[0.14, 0.18, 0.08] + [0.065] * (len(table_df.columns) - 3),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.5)

    # Save table as CSV for reference
    table_df.to_csv(output_dir / "run_configuration_table.csv", index=False)

    plt.tight_layout(rect=(0, 0, 1, 0.97))
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


def visualize_tokens_vs_time(df: pd.DataFrame, output_dir: Path):
    """Scatter plot of transcript tokens vs total time per video."""
    if "transcript_tokens" not in df.columns or df["transcript_tokens"].isna().all():
        _logger.warning("⚠️  No transcript token data available for token/time plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Transcript Tokens vs Total Time")
    ax.set_xlabel("Transcript Tokens")
    ax.set_ylabel("Total Time (seconds)")

    grouped = df.dropna(subset=["transcript_tokens", "total_time_seconds"]).groupby(
        "video_id"
    )
    for video_id, group in grouped:
        if group.empty:
            continue
        group_sorted = group.sort_values("total_time_seconds")
        label = video_id or group_sorted.iloc[0].get("video_title", "unknown")
        ax.plot(
            group_sorted["transcript_tokens"],
            group_sorted["total_time_seconds"],
            marker="o",
            label=label,
        )

    ax.legend(title="Video ID", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "05_tokens_vs_time.png", dpi=300, bbox_inches="tight")
    _logger.info(f"✅ Saved: {output_dir / '05_tokens_vs_time.png'}")
    plt.close()


def visualize_time_breakdown_table(df: pd.DataFrame, output_dir: Path):
    """Render a per-experiment pipeline breakdown table."""
    ordered_df = df.sort_values("total_time_seconds").reset_index(drop=True)
    if ordered_df.empty:
        return

    time_columns = [
        ("Experiment", "experiment"),
        ("Total (s)", "total_time_seconds"),
        ("Transcript (s)", "transcript_time"),
        ("Claims (s)", "claim_extraction_time"),
        ("Query Gen (s)", "query_generation_time"),
        ("Search API (s)", "google_search_time"),
        ("Reliability (s)", "reliability_time"),
        ("Content Fetch (s)", "content_fetch_time"),
        ("Evidence (s)", "evidence_extraction_time"),
        ("Output (s)", "output_generation_time"),
    ]

    table_data: dict[str, list[str | float]] = {}
    for label, column in time_columns:
        if column == "experiment":
            table_data[label] = ordered_df[column].astype(str).tolist()
            continue
        column_values = (
            ordered_df[column].fillna(0)
            if column in ordered_df
            else pd.Series([0] * len(ordered_df))
        )
        table_data[label] = column_values.astype(float).round(1).tolist()

    table_df = pd.DataFrame(table_data)
    table_df.insert(
        1, "Video ID", ordered_df.get("video_id", pd.Series([""] * len(ordered_df)))
    )

    fig_height = max(2.5, 0.4 * len(table_df) + 1)
    fig, ax = plt.subplots(figsize=(18, fig_height))
    ax.axis("off")
    ax.set_title(
        "Per-Experiment Pipeline Timing (seconds)", loc="left", fontweight="bold"
    )

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "04_pipeline_time_breakdown.png", dpi=300, bbox_inches="tight"
    )
    _logger.info(f"✅ Saved: {output_dir / '04_pipeline_time_breakdown.png'}")
    plt.close()

    table_df.to_csv(output_dir / "pipeline_time_breakdown.csv", index=False)


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

    # Enrich data with categorization
    _logger.info("🔄 Enriching experiment data...")
    df = enrich_experiment_data(df)

    # Load LLM call details for filtered runs
    run_ids = df["run_id"].tolist()
    llm_df = load_llm_calls_detail(runs_dir, run_ids)

    _logger.info(f"✅ Loaded {len(df)} experiment runs")
    _logger.info(f"✅ Loaded {len(llm_df)} LLM calls\n")

    # Generate visualizations
    _logger.info("📈 Generating visualizations...\n")

    # Core visualizations
    visualize_overview(df, output_dir)
    visualize_component_breakdown(llm_df, output_dir)
    visualize_claims_analysis(df, output_dir)
    visualize_time_breakdown_table(df, output_dir)
    visualize_tokens_vs_time(df, output_dir)

    # Phase 1 Global Quantitative visualizations
    _logger.info("\n📊 Generating Phase 1 analysis...\n")
    plot_ofat_analysis(df, output_dir)
    plot_strategy_comparison(df, output_dir)
    plot_scalability_analysis(df, output_dir)
    plot_cost_breakdown(df, llm_df, output_dir)
    plot_evidence_quality_analysis(df, output_dir)

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
