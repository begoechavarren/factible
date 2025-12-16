#!/usr/bin/env python3
import json
import typer
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

# LaTeX-style settings for academic figures
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "text.usetex": False,  # Set True if LaTeX is installed
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
    }
)

app = typer.Typer()


@app.command()
def main(
    run_id: str = typer.Option(
        ...,
        "--run-id",
        help="Run ID directory name (e.g., 20251214_002019_precision_recall_curve)",
    ),
):
    """Create precision-recall curve from evaluation results."""
    # Base directory - relative to this script location
    SCRIPT_DIR = Path(__file__).parent
    EVAL_RESULTS_DIR = SCRIPT_DIR.parent / "data" / "eval_results"
    BASE_DIR = EVAL_RESULTS_DIR / run_id

    if not BASE_DIR.exists():
        typer.echo(f"Error: Directory not found: {BASE_DIR}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Using run: {run_id}")
    typer.echo(f"Reading from: {BASE_DIR}\n")

    # Configuration directories
    configs = [1, 3, 5, 7, 10, 15]

    # Data storage
    data: dict = {
        "max_claims": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "map": [],
    }

    # Read aggregate reports
    for k in configs:
        # Find the aggregate report for this configuration
        config_dir = BASE_DIR / f"max_claims_{k}"

        # Find the timestamp subdirectory
        timestamp_dirs = list(config_dir.glob("*"))
        if not timestamp_dirs:
            typer.echo(f"Warning: No results found for max_claims={k}")
            continue

        # Use most recent
        report_path = timestamp_dirs[0] / "aggregate_report.json"

        if not report_path.exists():
            typer.echo(f"Warning: No aggregate report found for max_claims={k}")
            continue

        # Load the report
        with open(report_path) as f:
            report = json.load(f)

        # Extract metrics
        data["max_claims"].append(k)
        data["precision"].append(report["claim_extraction"]["precision_at_k_mean"])
        data["recall"].append(report["claim_extraction"]["recall_mean"])
        data["f1"].append(report["claim_extraction"]["f1_mean"])
        data["map"].append(report["claim_extraction"]["map_mean"])

        typer.echo(
            f"max_claims={k}: P={data['precision'][-1]:.3f}, R={data['recall'][-1]:.3f}, F1={data['f1'][-1]:.3f}, MAP={data['map'][-1]:.3f}"
        )

    typer.echo(f"\nLoaded data for {len(data['max_claims'])} configurations")

    # Create the precision-recall curve with academic styling
    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Plot the curve - darker color, cleaner style
    ax.plot(
        data["recall"],
        data["precision"],
        color="#2c3e50",
        marker="o",
        linestyle="-",
        linewidth=1.5,
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=1.5,
        label="Precision-Recall Curve",
    )

    # Annotate each point with max_claims value
    for i, k in enumerate(data["max_claims"]):
        # Highlight max_claims=5 (primary configuration)
        if k == 5:
            ax.annotate(
                f"k={k}",
                (data["recall"][i], data["precision"][i]),
                textcoords="offset points",
                xytext=(8, 8),
                ha="left",
                fontsize=10,
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="#f0f0f0",
                    edgecolor="#666666",
                    linewidth=0.5,
                ),
            )
        else:
            ax.annotate(
                f"k={k}",
                (data["recall"][i], data["precision"][i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
            )

    # Clean academic styling
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Tradeoff by max_claims (k)", pad=10)
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.set_xlim(0, 0.65)
    ax.set_ylim(0.65, 0.85)

    # Clean up spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add text box with operating point (top right corner)
    idx_5 = data["max_claims"].index(5)
    textstr = f"Operating point (k=5):\nPrecision: {data['precision'][idx_5]:.1%}\nRecall: {data['recall'][idx_5]:.1%}\nF1: {data['f1'][idx_5]:.1%}"
    props = dict(
        boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", linewidth=0.5
    )
    ax.text(
        0.97,
        0.97,
        textstr,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=props,
    )

    plt.tight_layout()

    # Save the figure in the run's directory
    output_dir = BASE_DIR
    output_path = output_dir / f"precision_recall_curve_{run_id}.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    typer.echo(f"\nPrecision-Recall curve saved to: {output_path}")

    # Also save as PNG
    output_png = output_path.with_suffix(".png")
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    typer.echo(f"PNG version saved to: {output_png}")

    # Create a summary table
    typer.echo("\n" + "=" * 80)
    typer.echo("PRECISION-RECALL TRADEOFF SUMMARY")
    typer.echo("=" * 80)
    typer.echo(
        f"{'max_claims':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'MAP':<12}"
    )
    typer.echo("-" * 80)
    for i, k in enumerate(data["max_claims"]):
        marker = " <- PRIMARY" if k == 5 else ""
        typer.echo(
            f"{k:<12} {data['precision'][i]:<12.3f} {data['recall'][i]:<12.3f} {data['f1'][i]:<12.3f} {data['map'][i]:<12.3f}{marker}"
        )
    typer.echo("=" * 80)

    # Save the table as JSON for easy access
    table_data = {
        "configurations": [
            {
                "max_claims": data["max_claims"][i],
                "precision": data["precision"][i],
                "recall": data["recall"][i],
                "f1": data["f1"][i],
                "map": data["map"][i],
                "is_primary": data["max_claims"][i] == 5,
            }
            for i in range(len(data["max_claims"]))
        ]
    }

    table_path = output_dir / f"precision_recall_summary_{run_id}.json"
    with open(table_path, "w") as f:
        json.dump(table_data, f, indent=2)
    typer.echo(f"\nSummary table saved to: {table_path}")


if __name__ == "__main__":
    app()
