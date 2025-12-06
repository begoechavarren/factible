"""
Phase 1 Visualization Functions for Global Quantitative Analysis.

These functions add OFAT analysis, strategy comparison, scalability analysis,
cost breakdown, and topic analysis to the experiment analysis pipeline.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec


def enrich_experiment_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich experiment data with categorization and tags.

    Adds:
    - experiment_type: baseline, ofat, strategic, custom
    - ofat_parameter: claims, queries, results (if OFAT)
    - ofat_value: the parameter value being tested
    - strategy_type: minimal, deep, broad (if strategic)
    - video_id: extracted from experiment name
    """
    df = df.copy()

    # Initialize new columns
    df["experiment_type"] = "custom"
    df["ofat_parameter"] = None
    df["ofat_value"] = None
    df["strategy_type"] = None
    df["video_id"] = None

    for idx, row in df.iterrows():
        exp_name = row["experiment"]

        # Extract video_id from experiment name
        # Format: vary_claims_claims1_fossil_fuels_greenest_energy
        # or baseline_fossil_fuels_greenest_energy
        parts = exp_name.split("_")

        # Find where the video id starts
        # For OFAT: skip "vary", param type, and value (e.g., "vary_claims_claims1")
        # For baseline: skip "baseline"
        # For strategic: skip strategy type
        if (
            "vary_claims" in exp_name
            or "vary_queries" in exp_name
            or "vary_results" in exp_name
        ):
            # OFAT format: vary_PARAM_PARAMVALUE_VIDEO_ID
            # Skip first 3 parts (vary, param type, param value)
            video_id = "_".join(parts[3:])
        elif "baseline" in exp_name:
            # Baseline format: baseline_VIDEO_ID
            video_id = "_".join(parts[1:])
        elif any(x in exp_name for x in ["minimal", "deep", "broad"]):
            # Strategic format: STRATEGY_VIDEO_ID
            video_id = "_".join(parts[1:])
        else:
            # Custom format - use everything
            video_id = exp_name

        df.at[idx, "video_id"] = video_id

        # Categorize baseline
        if "baseline" in exp_name.lower():
            df.at[idx, "experiment_type"] = "baseline"

        # Categorize OFAT experiments
        elif "vary_claims" in exp_name:
            df.at[idx, "experiment_type"] = "ofat"
            df.at[idx, "ofat_parameter"] = "max_claims"
            df.at[idx, "ofat_value"] = row["max_claims"]
        elif "vary_queries" in exp_name:
            df.at[idx, "experiment_type"] = "ofat"
            df.at[idx, "ofat_parameter"] = "max_queries_per_claim"
            df.at[idx, "ofat_value"] = row["max_queries_per_claim"]
        elif "vary_results" in exp_name:
            df.at[idx, "experiment_type"] = "ofat"
            df.at[idx, "ofat_parameter"] = "max_results_per_query"
            df.at[idx, "ofat_value"] = row["max_results_per_query"]

        # Categorize strategic experiments
        elif "minimal" in exp_name.lower():
            df.at[idx, "experiment_type"] = "strategic"
            df.at[idx, "strategy_type"] = "minimal"
        elif "deep" in exp_name.lower():
            df.at[idx, "experiment_type"] = "strategic"
            df.at[idx, "strategy_type"] = "deep"
        elif "broad" in exp_name.lower():
            df.at[idx, "experiment_type"] = "strategic"
            df.at[idx, "strategy_type"] = "broad"

    return df


def plot_ofat_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create OFAT (One Factor At a Time) sensitivity analysis plots.

    Generates line plots showing how each parameter affects:
    - Total time
    - Total cost
    - Total sources found
    - Average sources per claim
    """
    ofat_df = df[df["experiment_type"] == "ofat"].copy()

    if ofat_df.empty:
        print("⚠️  No OFAT experiments found. Skipping OFAT analysis.")
        return

    # Get unique parameters tested
    parameters = ofat_df["ofat_parameter"].dropna().unique()

    if len(parameters) == 0:
        print("⚠️  No OFAT parameters detected. Skipping OFAT analysis.")
        return

    # Create figure with subplots for each parameter
    n_params = len(parameters)
    fig = plt.figure(figsize=(18, 6 * n_params))

    # Collect all aggregated data for CSV export
    all_ofat_data = []

    for param_idx, param in enumerate(parameters):
        param_data = ofat_df[ofat_df["ofat_parameter"] == param].copy()

        # Group by parameter value and aggregate
        grouped = (
            param_data.groupby("ofat_value")
            .agg(
                {
                    "total_time_seconds": ["mean", "std"],
                    "llm_total_cost": ["mean", "std"],
                    "total_sources": ["mean", "std"],
                    "avg_sources_per_claim": ["mean", "std"],
                    "claims_processed": ["mean"],
                }
            )
            .reset_index()
        )

        # Flatten column names
        grouped.columns = ["_".join(col).strip("_") for col in grouped.columns.values]
        # The ofat_value column should be renamed to param_value
        # After flattening, it's just "ofat_value" (no trailing underscore)
        grouped = grouped.rename(columns={"ofat_value": "param_value"})
        grouped = grouped.sort_values("param_value")

        # Add parameter name to the data and collect for CSV export
        grouped_with_param = grouped.copy()
        grouped_with_param["parameter_name"] = param
        all_ofat_data.append(grouped_with_param)

        # Create 4 subplots for this parameter
        gs = GridSpec(
            2,
            2,
            figure=fig,
            left=0.08,
            right=0.95,
            top=0.92 - param_idx * (0.95 / n_params),
            bottom=0.08 + (n_params - param_idx - 1) * (0.95 / n_params),
            hspace=0.3,
            wspace=0.25,
        )

        # Format parameter name for display
        param_display = param.replace("_", " ").title()

        # Plot 1: Time vs Parameter
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(
            grouped["param_value"],
            grouped["total_time_seconds_mean"],
            marker="o",
            linewidth=2,
            markersize=8,
            color="#2E86AB",
        )
        if "total_time_seconds_std" in grouped.columns:
            ax1.fill_between(
                grouped["param_value"],
                grouped["total_time_seconds_mean"] - grouped["total_time_seconds_std"],
                grouped["total_time_seconds_mean"] + grouped["total_time_seconds_std"],
                alpha=0.2,
                color="#2E86AB",
            )
        ax1.set_xlabel(param_display, fontsize=11, fontweight="bold")
        ax1.set_ylabel("Total Time (seconds)", fontsize=11)
        ax1.set_title(
            f"Processing Time vs {param_display}", fontsize=12, fontweight="bold"
        )
        ax1.grid(True, alpha=0.3)

        # Add trend line
        z = np.polyfit(grouped["param_value"], grouped["total_time_seconds_mean"], 1)
        p = np.poly1d(z)
        ax1.plot(
            grouped["param_value"],
            p(grouped["param_value"]),
            "--",
            color="#A23B72",
            alpha=0.7,
            label=f"Trend: y={z[0]:.1f}x+{z[1]:.1f}",
        )
        ax1.legend(fontsize=9)

        # Plot 2: Cost vs Parameter
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(
            grouped["param_value"],
            grouped["llm_total_cost_mean"],
            marker="s",
            linewidth=2,
            markersize=8,
            color="#F18F01",
        )
        if "llm_total_cost_std" in grouped.columns:
            ax2.fill_between(
                grouped["param_value"],
                grouped["llm_total_cost_mean"] - grouped["llm_total_cost_std"],
                grouped["llm_total_cost_mean"] + grouped["llm_total_cost_std"],
                alpha=0.2,
                color="#F18F01",
            )
        ax2.set_xlabel(param_display, fontsize=11, fontweight="bold")
        ax2.set_ylabel("Total Cost (USD)", fontsize=11)
        ax2.set_title(f"Cost vs {param_display}", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3)

        # Plot 3: Total Sources vs Parameter
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(
            grouped["param_value"],
            grouped["total_sources_mean"],
            marker="^",
            linewidth=2,
            markersize=8,
            color="#06A77D",
        )
        if "total_sources_std" in grouped.columns:
            ax3.fill_between(
                grouped["param_value"],
                grouped["total_sources_mean"] - grouped["total_sources_std"],
                grouped["total_sources_mean"] + grouped["total_sources_std"],
                alpha=0.2,
                color="#06A77D",
            )
        ax3.set_xlabel(param_display, fontsize=11, fontweight="bold")
        ax3.set_ylabel("Total Evidence Sources", fontsize=11)
        ax3.set_title(
            f"Evidence Coverage vs {param_display}", fontsize=12, fontweight="bold"
        )
        ax3.grid(True, alpha=0.3)

        # Plot 4: Efficiency (sources per second) vs Parameter
        ax4 = fig.add_subplot(gs[1, 1])
        efficiency = grouped["total_sources_mean"] / grouped["total_time_seconds_mean"]
        ax4.plot(
            grouped["param_value"],
            efficiency,
            marker="D",
            linewidth=2,
            markersize=8,
            color="#D62828",
        )
        ax4.set_xlabel(param_display, fontsize=11, fontweight="bold")
        ax4.set_ylabel("Efficiency (sources/second)", fontsize=11)
        ax4.set_title(
            f"Search Efficiency vs {param_display}", fontsize=12, fontweight="bold"
        )
        ax4.grid(True, alpha=0.3)

        # Add annotations for optimal point
        max_eff_idx = efficiency.idxmax()
        optimal_value = grouped.loc[max_eff_idx, "param_value"]
        ax4.annotate(
            f"Peak efficiency\nat {param_display}={optimal_value}",
            xy=(optimal_value, efficiency.max()),
            xytext=(10, 20),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3"),
        )

    fig.suptitle("OFAT Sensitivity Analysis", fontsize=18, fontweight="bold", y=0.995)

    # Save aggregated data to CSV
    if all_ofat_data:
        ofat_csv_data = pd.concat(all_ofat_data, ignore_index=True)
        ofat_csv_data.to_csv(
            output_dir / "06_ofat_sensitivity_data_aggregated.csv", index=False
        )
        print("✅ Saved: 06_ofat_sensitivity_data_aggregated.csv")

    # Also save raw OFAT data with video_id for detailed analysis
    ofat_raw_columns = [
        "experiment",
        "video_id",
        "ofat_parameter",
        "ofat_value",
        "total_time_seconds",
        "llm_total_cost",
        "total_sources",
        "avg_sources_per_claim",
        "claims_processed",
        "claims_extracted",
    ]
    ofat_df[ofat_raw_columns].to_csv(
        output_dir / "06_ofat_sensitivity_data_raw.csv", index=False
    )
    print("✅ Saved: 06_ofat_sensitivity_data_raw.csv")

    plt.savefig(output_dir / "06_ofat_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ Generated: 06_ofat_sensitivity.png")


def plot_strategy_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Compare strategic configurations (minimal, deep, broad).

    Creates a dashboard comparing:
    - Processing time
    - Cost
    - Claims processed
    - Total sources
    - Confidence distribution
    """
    strategic_df = df[df["experiment_type"] == "strategic"].copy()

    if strategic_df.empty:
        print("⚠️  No strategic experiments found. Skipping strategy comparison.")
        return

    strategies = strategic_df["strategy_type"].dropna().unique()

    if len(strategies) == 0:
        print("⚠️  No strategies detected. Skipping strategy comparison.")
        return

    # Aggregate by strategy
    strategy_stats = (
        strategic_df.groupby("strategy_type")
        .agg(
            {
                "total_time_seconds": ["mean", "std"],
                "llm_total_cost": ["mean", "std"],
                "claims_processed": ["mean", "std"],
                "total_sources": ["mean", "std"],
                "high_confidence_count": ["mean"],
                "medium_confidence_count": ["mean"],
                "low_confidence_count": ["mean"],
            }
        )
        .reset_index()
    )

    # Flatten columns
    strategy_stats.columns = [
        "_".join(col).strip("_") for col in strategy_stats.columns.values
    ]
    strategy_stats = strategy_stats.rename(columns={"strategy_type_": "strategy"})

    # Create figure
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Define colors for strategies
    colors = {"minimal": "#06A77D", "deep": "#2E86AB", "broad": "#D62828"}
    strategy_stats["color"] = strategy_stats["strategy"].map(colors)

    # Plot 1: Processing Time
    ax1 = fig.add_subplot(gs[0, 0])
    bars1 = ax1.bar(
        strategy_stats["strategy"],
        strategy_stats["total_time_seconds_mean"],
        yerr=strategy_stats.get("total_time_seconds_std", 0),
        color=strategy_stats["color"],
        alpha=0.8,
        capsize=5,
    )
    ax1.set_ylabel("Time (seconds)", fontsize=11, fontweight="bold")
    ax1.set_title("Processing Time by Strategy", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}s",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Plot 2: Cost
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.bar(
        strategy_stats["strategy"],
        strategy_stats["llm_total_cost_mean"],
        yerr=strategy_stats.get("llm_total_cost_std", 0),
        color=strategy_stats["color"],
        alpha=0.8,
        capsize=5,
    )
    ax2.set_ylabel("Cost (USD)", fontsize=11, fontweight="bold")
    ax2.set_title("Cost by Strategy", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    for bar in bars2:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"${height:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Plot 3: Claims Processed
    ax3 = fig.add_subplot(gs[0, 2])
    bars3 = ax3.bar(
        strategy_stats["strategy"],
        strategy_stats["claims_processed_mean"],
        yerr=strategy_stats.get("claims_processed_std", 0),
        color=strategy_stats["color"],
        alpha=0.8,
        capsize=5,
    )
    ax3.set_ylabel("Claims Processed", fontsize=11, fontweight="bold")
    ax3.set_title("Claims Processed by Strategy", fontsize=12, fontweight="bold")
    ax3.grid(axis="y", alpha=0.3)

    for bar in bars3:
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Plot 4: Total Sources
    ax4 = fig.add_subplot(gs[1, 0])
    bars4 = ax4.bar(
        strategy_stats["strategy"],
        strategy_stats["total_sources_mean"],
        yerr=strategy_stats.get("total_sources_std", 0),
        color=strategy_stats["color"],
        alpha=0.8,
        capsize=5,
    )
    ax4.set_ylabel("Total Evidence Sources", fontsize=11, fontweight="bold")
    ax4.set_title("Evidence Coverage by Strategy", fontsize=12, fontweight="bold")
    ax4.grid(axis="y", alpha=0.3)

    for bar in bars4:
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Plot 5: Confidence Distribution
    ax5 = fig.add_subplot(gs[1, 1])
    x = np.arange(len(strategy_stats))
    width = 0.25

    ax5.bar(
        x - width,
        strategy_stats["high_confidence_count_mean"],
        width,
        label="High",
        color="#06A77D",
        alpha=0.8,
    )
    ax5.bar(
        x,
        strategy_stats["medium_confidence_count_mean"],
        width,
        label="Medium",
        color="#F18F01",
        alpha=0.8,
    )
    ax5.bar(
        x + width,
        strategy_stats["low_confidence_count_mean"],
        width,
        label="Low",
        color="#D62828",
        alpha=0.8,
    )

    ax5.set_ylabel("Count", fontsize=11, fontweight="bold")
    ax5.set_title("Confidence Distribution by Strategy", fontsize=12, fontweight="bold")
    ax5.set_xticks(x)
    ax5.set_xticklabels(strategy_stats["strategy"])
    ax5.legend()
    ax5.grid(axis="y", alpha=0.3)

    # Plot 6: Cost-Time Bubble Chart
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.scatter(
        strategy_stats["total_time_seconds_mean"],
        strategy_stats["llm_total_cost_mean"],
        s=strategy_stats["total_sources_mean"] * 10,  # Size by sources
        c=[colors[s] for s in strategy_stats["strategy"]],
        alpha=0.6,
        edgecolors="black",
        linewidth=2,
    )

    # Add labels
    for idx, row in strategy_stats.iterrows():
        ax6.annotate(
            row["strategy"],
            (row["total_time_seconds_mean"], row["llm_total_cost_mean"]),
            fontsize=10,
            fontweight="bold",
            xytext=(5, 5),
            textcoords="offset points",
        )

    ax6.set_xlabel("Time (seconds)", fontsize=11, fontweight="bold")
    ax6.set_ylabel("Cost (USD)", fontsize=11, fontweight="bold")
    ax6.set_title(
        "Cost-Time Trade-off (bubble size = sources)", fontsize=12, fontweight="bold"
    )
    ax6.grid(True, alpha=0.3)

    fig.suptitle("Strategic Configuration Comparison", fontsize=18, fontweight="bold")

    # Save aggregated data to CSV
    strategy_stats.to_csv(
        output_dir / "07_strategy_comparison_data_aggregated.csv", index=False
    )
    print("✅ Saved: 07_strategy_comparison_data_aggregated.csv")

    # Also save raw strategic data with video_id
    strategic_raw_columns = [
        "experiment",
        "video_id",
        "strategy_type",
        "total_time_seconds",
        "llm_total_cost",
        "claims_processed",
        "total_sources",
        "high_confidence_count",
        "medium_confidence_count",
        "low_confidence_count",
    ]
    strategic_df[strategic_raw_columns].to_csv(
        output_dir / "07_strategy_comparison_data_raw.csv", index=False
    )
    print("✅ Saved: 07_strategy_comparison_data_raw.csv")

    plt.savefig(output_dir / "07_strategy_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ Generated: 07_strategy_comparison.png")


def plot_scalability_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Analyze relationship between video properties and system performance.

    Creates scatter plots with regression lines for:
    - Video duration vs processing time
    - Transcript length vs cost
    - Claims extracted vs video duration
    """
    # Filter successful runs with valid data
    valid_df = df[
        df["success"]
        & (df["video_duration_seconds"].notna())
        & (df["video_duration_seconds"] > 0)
    ].copy()

    if len(valid_df) < 3:
        print(
            "⚠️  Insufficient data for scalability analysis. Need at least 3 runs with video duration."
        )
        return

    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 3, figure=fig, hspace=0.3, wspace=0.25)

    # Plot 1: Video Duration vs Processing Time
    ax1 = fig.add_subplot(gs[0, 0])
    x1 = valid_df["video_duration_seconds"]
    y1 = valid_df["total_time_seconds"]

    ax1.scatter(
        x1, y1, alpha=0.6, s=100, color="#2E86AB", edgecolors="black", linewidth=1
    )

    # Add regression line
    if len(x1) >= 2:
        z1 = np.polyfit(x1, y1, 1)
        p1 = np.poly1d(z1)
        x_line = np.linspace(x1.min(), x1.max(), 100)
        ax1.plot(
            x_line,
            p1(x_line),
            "--",
            color="#D62828",
            linewidth=2,
            label=f"y = {z1[0]:.2f}x + {z1[1]:.1f}",
        )

        # Calculate R²
        r_squared = np.corrcoef(x1, y1)[0, 1] ** 2
        ax1.text(
            0.05,
            0.95,
            f"R² = {r_squared:.3f}",
            transform=ax1.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    ax1.set_xlabel("Video Duration (seconds)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Processing Time (seconds)", fontsize=11, fontweight="bold")
    ax1.set_title(
        "Scalability: Duration vs Processing Time", fontsize=12, fontweight="bold"
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Transcript Length vs Cost
    ax2 = fig.add_subplot(gs[0, 1])
    valid_cost = valid_df[valid_df["transcript_length"].notna()]

    if len(valid_cost) >= 2:
        x2 = valid_cost["transcript_length"]
        y2 = valid_cost["llm_total_cost"]

        ax2.scatter(
            x2, y2, alpha=0.6, s=100, color="#F18F01", edgecolors="black", linewidth=1
        )

        # Regression
        z2 = np.polyfit(x2, y2, 1)
        p2 = np.poly1d(z2)
        x_line2 = np.linspace(x2.min(), x2.max(), 100)
        ax2.plot(
            x_line2,
            p2(x_line2),
            "--",
            color="#D62828",
            linewidth=2,
            label=f"y = {z2[0]:.2e}x + {z2[1]:.3f}",
        )

        r_squared2 = np.corrcoef(x2, y2)[0, 1] ** 2
        ax2.text(
            0.05,
            0.95,
            f"R² = {r_squared2:.3f}",
            transform=ax2.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        ax2.set_xlabel("Transcript Length (characters)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Total Cost (USD)", fontsize=11, fontweight="bold")
        ax2.set_title(
            "Cost Scaling with Transcript Length", fontsize=12, fontweight="bold"
        )
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # Plot 3: Claims Extracted vs Video Duration
    ax3 = fig.add_subplot(gs[0, 2])
    valid_claims = valid_df[valid_df["claims_extracted"] > 0]

    if len(valid_claims) >= 2:
        x3 = valid_claims["video_duration_seconds"]
        y3 = valid_claims["claims_extracted"]

        ax3.scatter(
            x3, y3, alpha=0.6, s=100, color="#06A77D", edgecolors="black", linewidth=1
        )

        # Regression
        z3 = np.polyfit(x3, y3, 1)
        p3 = np.poly1d(z3)
        x_line3 = np.linspace(x3.min(), x3.max(), 100)
        ax3.plot(
            x_line3,
            p3(x_line3),
            "--",
            color="#D62828",
            linewidth=2,
            label=f"y = {z3[0]:.3f}x + {z3[1]:.1f}",
        )

        r_squared3 = np.corrcoef(x3, y3)[0, 1] ** 2
        ax3.text(
            0.05,
            0.95,
            f"R² = {r_squared3:.3f}",
            transform=ax3.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        ax3.set_xlabel("Video Duration (seconds)", fontsize=11, fontweight="bold")
        ax3.set_ylabel("Claims Extracted", fontsize=11, fontweight="bold")
        ax3.set_title("Claim Density vs Video Duration", fontsize=12, fontweight="bold")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    fig.suptitle("Scalability Analysis", fontsize=18, fontweight="bold")

    # Save scalability data to CSV
    scalability_data = valid_df[
        [
            "experiment",
            "video_id",
            "video_duration_seconds",
            "total_time_seconds",
            "transcript_length",
            "llm_total_cost",
            "claims_extracted",
            "claims_processed",
        ]
    ].copy()
    scalability_data.to_csv(
        output_dir / "08_scalability_analysis_data.csv", index=False
    )
    print("✅ Saved: 08_scalability_analysis_data.csv")

    plt.savefig(
        output_dir / "08_scalability_analysis.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    print("✅ Generated: 08_scalability_analysis.png")


def plot_cost_breakdown(
    df: pd.DataFrame, llm_df: pd.DataFrame, output_dir: Path
) -> None:
    """
    Detailed cost analysis and breakdown by component.

    Creates:
    - Pie chart of cost by component
    - Cost per claim comparison
    - Cumulative cost pattern
    """
    if llm_df.empty:
        print("⚠️  No LLM call data available. Skipping cost breakdown.")
        return

    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 3, figure=fig, hspace=0.3, wspace=0.25)

    # Plot 1: Cost Breakdown by Component (Pie Chart)
    ax1 = fig.add_subplot(gs[0, 0])

    # Aggregate cost by component
    component_cost = llm_df.groupby("component")["cost_usd"].sum()

    if component_cost.sum() > 0:
        colors_pie = ["#2E86AB", "#F18F01", "#06A77D", "#D62828", "#A23B72"]
        wedges, texts, autotexts = ax1.pie(
            component_cost,
            labels=component_cost.index,
            autopct="%1.1f%%",
            colors=colors_pie[: len(component_cost)],
            startangle=90,
            textprops={"fontsize": 10, "fontweight": "bold"},
        )
        ax1.set_title("Cost Distribution by Component", fontsize=12, fontweight="bold")
    else:
        ax1.text(
            0.5,
            0.5,
            "No cost data\n(using free models)",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax1.transAxes,
        )
        ax1.set_title("Cost Distribution by Component", fontsize=12, fontweight="bold")

    # Plot 2: Cost per Claim by Experiment
    ax2 = fig.add_subplot(gs[0, 1])

    # Calculate cost per claim
    cost_per_claim = df[df["claims_processed"] > 0].copy()
    cost_per_claim["cost_per_claim"] = (
        cost_per_claim["llm_total_cost"] / cost_per_claim["claims_processed"]
    )

    if not cost_per_claim.empty and cost_per_claim["cost_per_claim"].sum() > 0:
        cost_per_claim = cost_per_claim.sort_values(
            "cost_per_claim", ascending=False
        ).head(10)

        bars = ax2.barh(
            range(len(cost_per_claim)),
            cost_per_claim["cost_per_claim"],
            color="#F18F01",
            alpha=0.8,
        )
        ax2.set_yticks(range(len(cost_per_claim)))
        ax2.set_yticklabels(
            [
                exp[:30] + "..." if len(exp) > 30 else exp
                for exp in cost_per_claim["experiment"]
            ],
            fontsize=9,
        )
        ax2.set_xlabel("Cost per Claim (USD)", fontsize=11, fontweight="bold")
        ax2.set_title(
            "Top 10: Cost Efficiency by Experiment", fontsize=12, fontweight="bold"
        )
        ax2.grid(axis="x", alpha=0.3)

        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(
                width,
                bar.get_y() + bar.get_height() / 2.0,
                f"${width:.4f}",
                ha="left",
                va="center",
                fontsize=9,
            )
    else:
        ax2.text(
            0.5,
            0.5,
            "No cost data available",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax2.transAxes,
        )
        ax2.set_title("Cost per Claim by Experiment", fontsize=12, fontweight="bold")

    # Plot 3: Token Distribution
    ax3 = fig.add_subplot(gs[0, 2])

    if (
        "input_tokens_estimated" in llm_df.columns
        and "output_tokens_estimated" in llm_df.columns
    ):
        component_tokens = llm_df.groupby("component").agg(
            {"input_tokens_estimated": "sum", "output_tokens_estimated": "sum"}
        )

        x = np.arange(len(component_tokens))
        width = 0.35

        ax3.bar(
            x - width / 2,
            component_tokens["input_tokens_estimated"],
            width,
            label="Input Tokens",
            color="#2E86AB",
            alpha=0.8,
        )
        ax3.bar(
            x + width / 2,
            component_tokens["output_tokens_estimated"],
            width,
            label="Output Tokens",
            color="#F18F01",
            alpha=0.8,
        )

        ax3.set_ylabel("Tokens", fontsize=11, fontweight="bold")
        ax3.set_title("Token Usage by Component", fontsize=12, fontweight="bold")
        ax3.set_xticks(x)
        ax3.set_xticklabels(component_tokens.index, rotation=45, ha="right")
        ax3.legend()
        ax3.grid(axis="y", alpha=0.3)
    else:
        ax3.text(
            0.5,
            0.5,
            "Token data not available",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax3.transAxes,
        )
        ax3.set_title("Token Usage by Component", fontsize=12, fontweight="bold")

    fig.suptitle("Cost Analysis", fontsize=18, fontweight="bold")

    # Save cost breakdown data to CSV
    # Component-level costs
    component_cost_df = component_cost.reset_index()
    component_cost_df.columns = ["component", "total_cost_usd"]
    component_cost_df.to_csv(
        output_dir / "09_cost_breakdown_by_component.csv", index=False
    )
    print("✅ Saved: 09_cost_breakdown_by_component.csv")

    # Cost per claim data (if exists)
    if not cost_per_claim.empty:
        cost_per_claim[
            [
                "experiment",
                "video_id",
                "claims_processed",
                "llm_total_cost",
                "cost_per_claim",
            ]
        ].to_csv(output_dir / "09_cost_per_claim.csv", index=False)
        print("✅ Saved: 09_cost_per_claim.csv")

    plt.savefig(output_dir / "09_cost_breakdown.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ Generated: 09_cost_breakdown.png")


def plot_evidence_quality_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Analyze evidence quality across experiments.

    Creates:
    - Stance distribution by experiment
    - Confidence distribution
    - Sources per claim distribution
    - Reliability score patterns
    """
    # Filter valid data
    valid_df = df[df["success"] & (df["claims_processed"] > 0)].copy()

    if len(valid_df) < 2:
        print(
            "⚠️  Insufficient data for evidence quality analysis. Need at least 2 successful runs."
        )
        return

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Plot 1: Stance Distribution Stacked Bar
    ax1 = fig.add_subplot(gs[0, 0])

    stance_cols = ["supports_count", "refutes_count", "mixed_count", "unclear_count"]
    stance_data = valid_df[["experiment"] + stance_cols].copy()

    # Limit to top 10 experiments by total claims
    stance_data["total"] = stance_data[stance_cols].sum(axis=1)
    stance_data = stance_data.nlargest(10, "total")

    x = np.arange(len(stance_data))
    width = 0.8

    # Create stacked bars
    colors_stance = {
        "supports": "#06A77D",
        "refutes": "#D62828",
        "mixed": "#F18F01",
        "unclear": "#6C757D",
    }

    bottom = np.zeros(len(stance_data))
    for col, label in [
        ("supports_count", "supports"),
        ("refutes_count", "refutes"),
        ("mixed_count", "mixed"),
        ("unclear_count", "unclear"),
    ]:
        values = stance_data[col].fillna(0).values
        ax1.bar(
            x,
            values,
            width,
            label=label.title(),
            bottom=bottom,
            color=colors_stance[label],
            alpha=0.85,
        )
        bottom += values

    ax1.set_ylabel("Count", fontsize=11, fontweight="bold")
    ax1.set_title("Evidence Stance Distribution", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [
            exp[:20] + "..." if len(exp) > 20 else exp
            for exp in stance_data["experiment"]
        ],
        rotation=45,
        ha="right",
        fontsize=9,
    )
    ax1.legend(loc="upper right")
    ax1.grid(axis="y", alpha=0.3)

    # Plot 2: Confidence Distribution
    ax2 = fig.add_subplot(gs[0, 1])

    conf_cols = [
        "high_confidence_count",
        "medium_confidence_count",
        "low_confidence_count",
    ]
    conf_data = valid_df[["experiment"] + conf_cols].copy()
    conf_data["total"] = conf_data[conf_cols].sum(axis=1)
    conf_data = conf_data.nlargest(10, "total")

    x2 = np.arange(len(conf_data))
    bottom2 = np.zeros(len(conf_data))

    colors_conf = {"high": "#06A77D", "medium": "#F18F01", "low": "#D62828"}

    for col, label in [
        ("high_confidence_count", "high"),
        ("medium_confidence_count", "medium"),
        ("low_confidence_count", "low"),
    ]:
        values = conf_data[col].fillna(0).values
        ax2.bar(
            x2,
            values,
            width,
            label=label.title(),
            bottom=bottom2,
            color=colors_conf[label],
            alpha=0.85,
        )
        bottom2 += values

    ax2.set_ylabel("Count", fontsize=11, fontweight="bold")
    ax2.set_title("Verdict Confidence Distribution", fontsize=12, fontweight="bold")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(
        [exp[:20] + "..." if len(exp) > 20 else exp for exp in conf_data["experiment"]],
        rotation=45,
        ha="right",
        fontsize=9,
    )
    ax2.legend(loc="upper right")
    ax2.grid(axis="y", alpha=0.3)

    # Plot 3: Sources per Claim Distribution (Box Plot)
    ax3 = fig.add_subplot(gs[0, 2])

    sources_data = valid_df[valid_df["avg_sources_per_claim"] > 0].copy()

    if len(sources_data) >= 2:
        # Group by experiment type if available
        if "experiment_type" in sources_data.columns:
            exp_types = sources_data["experiment_type"].unique()
            data_by_type = [
                sources_data[sources_data["experiment_type"] == et][
                    "avg_sources_per_claim"
                ].values
                for et in exp_types
            ]

            ax3.boxplot(
                data_by_type,
                labels=exp_types,
                patch_artist=True,
                boxprops=dict(facecolor="#2E86AB", alpha=0.7),
                medianprops=dict(color="#D62828", linewidth=2),
            )
            ax3.set_xlabel("Experiment Type", fontsize=11, fontweight="bold")
        else:
            ax3.boxplot(
                [sources_data["avg_sources_per_claim"].values],
                labels=["All"],
                patch_artist=True,
                boxprops=dict(facecolor="#2E86AB", alpha=0.7),
                medianprops=dict(color="#D62828", linewidth=2),
            )

        ax3.set_ylabel("Avg Sources per Claim", fontsize=11, fontweight="bold")
        ax3.set_title("Evidence Coverage Distribution", fontsize=12, fontweight="bold")
        ax3.grid(axis="y", alpha=0.3)

    # Plot 4: Total Sources vs Claims Processed (Scatter)
    ax4 = fig.add_subplot(gs[1, 0])

    scatter_data = valid_df[
        (valid_df["total_sources"] > 0) & (valid_df["claims_processed"] > 0)
    ].copy()

    if len(scatter_data) >= 2:
        ax4.scatter(
            scatter_data["claims_processed"],
            scatter_data["total_sources"],
            s=100,
            alpha=0.6,
            color="#2E86AB",
            edgecolors="black",
            linewidth=1,
        )

        # Add regression line
        x_scatter = scatter_data["claims_processed"].values
        y_scatter = scatter_data["total_sources"].values

        if len(x_scatter) >= 2:
            z = np.polyfit(x_scatter, y_scatter, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x_scatter.min(), x_scatter.max(), 100)
            ax4.plot(
                x_line,
                p(x_line),
                "--",
                color="#D62828",
                linewidth=2,
                label=f"y = {z[0]:.1f}x + {z[1]:.1f}",
            )

            # R²
            r_squared = np.corrcoef(x_scatter, y_scatter)[0, 1] ** 2
            ax4.text(
                0.05,
                0.95,
                f"R² = {r_squared:.3f}",
                transform=ax4.transAxes,
                fontsize=11,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        ax4.set_xlabel("Claims Processed", fontsize=11, fontweight="bold")
        ax4.set_ylabel("Total Sources Found", fontsize=11, fontweight="bold")
        ax4.set_title("Evidence Collection Efficiency", fontsize=12, fontweight="bold")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    # Plot 5: Stance Percentages (Pie Chart)
    ax5 = fig.add_subplot(gs[1, 1])

    total_stances = {
        "Supports": valid_df["supports_count"].sum(),
        "Refutes": valid_df["refutes_count"].sum(),
        "Mixed": valid_df["mixed_count"].sum(),
        "Unclear": valid_df["unclear_count"].sum(),
    }

    # Filter out zero values
    total_stances = {k: v for k, v in total_stances.items() if v > 0}

    if total_stances:
        colors_pie = [colors_stance[k.lower()] for k in total_stances.keys()]
        wedges, texts, autotexts = ax5.pie(
            total_stances.values(),
            labels=total_stances.keys(),
            autopct="%1.1f%%",
            colors=colors_pie,
            startangle=90,
            textprops={"fontsize": 10, "fontweight": "bold"},
        )
        ax5.set_title("Overall Stance Distribution", fontsize=12, fontweight="bold")

    # Plot 6: Average Sources by Stance
    ax6 = fig.add_subplot(gs[1, 2])

    # Calculate average sources for each stance type
    stance_sources = []
    stance_labels = []

    for stance, count_col in [
        ("Supports", "supports_count"),
        ("Refutes", "refutes_count"),
        ("Mixed", "mixed_count"),
        ("Unclear", "unclear_count"),
    ]:
        data_with_stance = valid_df[valid_df[count_col] > 0]
        if len(data_with_stance) > 0:
            avg = data_with_stance["avg_sources_per_claim"].mean()
            stance_sources.append(avg)
            stance_labels.append(stance)

    if stance_sources:
        bars = ax6.bar(
            stance_labels,
            stance_sources,
            color=[colors_stance[s.lower()] for s in stance_labels],
            alpha=0.85,
            edgecolor="black",
            linewidth=1.5,
        )

        ax6.set_ylabel("Avg Sources per Claim", fontsize=11, fontweight="bold")
        ax6.set_title("Evidence Depth by Stance", fontsize=12, fontweight="bold")
        ax6.grid(axis="y", alpha=0.3)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax6.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    fig.suptitle("Evidence Quality Analysis", fontsize=18, fontweight="bold")

    # Save evidence quality data to CSV
    evidence_quality_data = valid_df[
        [
            "experiment",
            "video_id",
            "claims_processed",
            "supports_count",
            "refutes_count",
            "mixed_count",
            "unclear_count",
            "high_confidence_count",
            "medium_confidence_count",
            "low_confidence_count",
            "avg_sources_per_claim",
            "total_sources",
        ]
    ].copy()
    evidence_quality_data.to_csv(
        output_dir / "10_evidence_quality_data.csv", index=False
    )
    print("✅ Saved: 10_evidence_quality_data.csv")

    plt.savefig(output_dir / "10_evidence_quality.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ Generated: 10_evidence_quality.png")
