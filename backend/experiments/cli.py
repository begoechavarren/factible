import logging

import typer
from dotenv import load_dotenv

from experiments.commands.evaluate import evaluate_command
from experiments.commands.precision_recall import precision_recall_command
from experiments.commands.run import run_command

app = typer.Typer(
    help="Factible Experiments - Run and analyze fact-checking experiments",
    no_args_is_help=True,
)

app.command(name="run", help="Run fact-checking experiments")(run_command)
app.command(name="evaluate", help="Evaluate experiment runs against ground truth")(
    evaluate_command
)
app.command(
    name="precision-recall",
    help="Generate precision-recall curve from evaluation results",
)(precision_recall_command)


@app.callback()
def main(
    log_level: str = typer.Option("INFO", help="Logging level"),
):
    """Run and evaluate experiments."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_dotenv()


if __name__ == "__main__":
    app()
