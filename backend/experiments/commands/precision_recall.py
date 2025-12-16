import json
import logging
import re
from pathlib import Path
from typing import Annotated

import matplotlib
import typer

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt

_logger = logging.getLogger(__name__)


def _extract_max_claims(dirname: str) -> int:
    """Extract max_claims value from directory name like 'max_claims_5'."""
    match = re.search(r"max_claims_(\d+)", dirname)
    if not match:
        raise ValueError(f"Invalid directory name: {dirname}")
    return int(match.group(1))


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


def _create_precision_recall_curve(eval_dir: Path, primary_k: int = 5) -> Path:
    """
    Create precision-recall curve from evaluation results.

    Args:
        eval_dir: Path to eval_results directory containing max_claims_* subdirs
        primary_k: The primary/operating point max_claims value to highlight

    Returns:
        Path to the generated PNG file
    """
    if not eval_dir.exists():
        raise FileNotFoundError(f"Directory not found: {eval_dir}")

    _logger.info(f"Reading from: {eval_dir}\n")

    # Auto-detect max_claims_* directories
    max_claims_dirs = sorted(
        [
            d
            for d in eval_dir.iterdir()
            if d.is_dir() and d.name.startswith("max_claims_")
        ],
        key=lambda d: _extract_max_claims(d.name),
    )

    if not max_claims_dirs:
        raise ValueError(f"No max_claims_* directories found in {eval_dir}")

    # Extract the k values
    configs = [_extract_max_claims(d.name) for d in max_claims_dirs]
    _logger.info(f"Found configurations: {configs}")

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
        config_dir = eval_dir / f"max_claims_{k}"

        # Find the timestamp subdirectory (use most recent if multiple)
        timestamp_dirs = sorted(
            [d for d in config_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
            reverse=True,
        )
        if not timestamp_dirs:
            _logger.warning(f"No results found for max_claims={k}")
            continue

        report_path = timestamp_dirs[0] / "aggregate_report.json"

        if not report_path.exists():
            _logger.warning(f"No aggregate report found for max_claims={k}")
            continue

        with open(report_path) as f:
            report = json.load(f)

        # Extract metrics
        data["max_claims"].append(k)
        data["precision"].append(report["claim_extraction"]["precision_at_k_mean"])
        data["recall"].append(report["claim_extraction"]["recall_mean"])
        data["f1"].append(report["claim_extraction"]["f1_mean"])
        data["map"].append(report["claim_extraction"]["map_mean"])

        _logger.info(
            f"max_claims={k}: P={data['precision'][-1]:.3f}, R={data['recall'][-1]:.3f}, "
            f"F1={data['f1'][-1]:.3f}, MAP={data['map'][-1]:.3f}"
        )

    _logger.info(f"\nLoaded data for {len(data['max_claims'])} configurations")

    if len(data["max_claims"]) < 2:
        raise ValueError("Need at least 2 data points to create a curve")

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
        # Highlight primary configuration
        if k == primary_k:
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

    # Add text box with operating point (top right corner) if primary_k exists
    if primary_k in data["max_claims"]:
        idx_primary = data["max_claims"].index(primary_k)
        textstr = f"Operating point (k={primary_k}):\nPrecision: {data['precision'][idx_primary]:.1%}\nRecall: {data['recall'][idx_primary]:.1%}\nF1: {data['f1'][idx_primary]:.1%}"
    else:
        # Use the middle point if primary_k not found
        idx_primary = len(data["max_claims"]) // 2
        textstr = f"k={data['max_claims'][idx_primary]}:\nPrecision: {data['precision'][idx_primary]:.1%}\nRecall: {data['recall'][idx_primary]:.1%}\nF1: {data['f1'][idx_primary]:.1%}"
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

    # Save the figure as PNG
    run_name = eval_dir.name
    output_path = eval_dir / f"precision_recall_curve_{run_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    _logger.info(f"\nPrecision-Recall curve saved to: {output_path}")

    plt.close()

    # Create a summary table
    _logger.info("\n" + "=" * 80)
    _logger.info("PRECISION-RECALL TRADEOFF SUMMARY")
    _logger.info("=" * 80)
    _logger.info(
        f"{'max_claims':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'MAP':<12}"
    )
    _logger.info("-" * 80)
    for i, k in enumerate(data["max_claims"]):
        marker = " <- PRIMARY" if k == primary_k else ""
        _logger.info(
            f"{k:<12} {data['precision'][i]:<12.3f} {data['recall'][i]:<12.3f} "
            f"{data['f1'][i]:<12.3f} {data['map'][i]:<12.3f}{marker}"
        )
    _logger.info("=" * 80)

    # Save the table as JSON
    table_data = {
        "primary_k": primary_k,
        "configurations": [
            {
                "max_claims": data["max_claims"][i],
                "precision": data["precision"][i],
                "recall": data["recall"][i],
                "f1": data["f1"][i],
                "map": data["map"][i],
                "is_primary": data["max_claims"][i] == primary_k,
            }
            for i in range(len(data["max_claims"]))
        ],
    }

    table_path = eval_dir / f"precision_recall_summary_{run_name}.json"
    with open(table_path, "w") as f:
        json.dump(table_data, f, indent=2)
    _logger.info(f"\nSummary table saved to: {table_path}")

    return output_path


def precision_recall_command(
    eval_dir: Annotated[
        str,
        typer.Option(
            help="Path to eval_results directory containing max_claims_* subdirs"
        ),
    ],
    primary_k: Annotated[
        int,
        typer.Option(help="Primary/operating point max_claims value to highlight"),
    ] = 5,
):
    """
    Generate precision-recall curve from evaluation results.

    The eval_dir should contain subdirectories named max_claims_1, max_claims_3, etc.,
    each with evaluation results from the evaluate command.

    Examples:
        factible-experiments precision-recall --eval-dir experiments/data/eval_results/my_experiment
    """
    eval_path = Path(eval_dir)
    _create_precision_recall_curve(eval_path, primary_k=primary_k)
