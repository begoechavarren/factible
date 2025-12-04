import json
import logging
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

_logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Track experiments with structured output to disk."""

    _current: Optional["ExperimentTracker"] = None

    def __init__(
        self,
        component: str,
        experiment_name: str,
        config: dict[str, Any],
        base_dir: Path = Path("factible/experiments/runs"),
    ):
        self.component = component
        self.experiment_name = experiment_name
        self.config = config

        # Create unique run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{timestamp}_{component}_{experiment_name}"
        self.run_dir = base_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Storage
        self.timing_data: dict[str, float] = {}
        self.metrics: dict[str, Any] = {}
        self.inputs: dict[str, Any] = {}
        self.outputs: dict[str, Any] = {}
        self.pydantic_calls: list[dict[str, Any]] = []

        # Context for tracing (claim/query being processed)
        self.context: dict[str, Any] = {}

        # Timing
        self.start_time = time.time()

        _logger.info(f"📊 Experiment started: {self.run_id}")

    @classmethod
    def get_current(cls) -> Optional["ExperimentTracker"]:
        """Get the current active tracker (singleton pattern)."""
        return cls._current

    @classmethod
    def set_current(cls, tracker: Optional["ExperimentTracker"]):
        """Set the current active tracker."""
        cls._current = tracker

    def log_timing(self, step: str, duration: float):
        """Log execution time for a step."""
        self.timing_data[step] = duration

    def log_metric(self, name: str, value: Any):
        """Log a metric value."""
        self.metrics[name] = value

    def log_input(self, name: str, data: Any):
        """Log input data."""
        self.inputs[name] = data

    def log_output(self, name: str, data: Any):
        """Log output data."""
        self.outputs[name] = data

    def log_pydantic_call(self, call_data: dict[str, Any]):
        """Log a Pydantic AI call."""
        # Include current context for traceability
        if self.context:
            call_data["context"] = self.context.copy()
        self.pydantic_calls.append(call_data)

    def set_context(self, **kwargs):
        """Set context for tracing (e.g., claim_index, claim_text, query_index)."""
        self.context.update(kwargs)

    def clear_context(self):
        """Clear the current context."""
        self.context.clear()

    def save(self):
        """Save all tracked data to disk."""
        total_time = time.time() - self.start_time

        # Save config (includes inputs)
        config_data = {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "component": self.component,
            "timestamp": datetime.now().isoformat(),
            **self.config,
            **self.inputs,  # Merge inputs into config
        }
        self._write_json("config.json", config_data)

        # Save LLM calls
        self._write_json("llm_calls.json", self.pydantic_calls)

        # Save outputs
        self._write_json("outputs.json", self.outputs)

        # Save transcript as separate text file (if available)
        if "final_output" in self.outputs:
            final_output = self.outputs["final_output"]
            if isinstance(final_output, dict) and "transcript_data" in final_output:
                transcript_data = final_output["transcript_data"]
                if isinstance(transcript_data, dict) and "text" in transcript_data:
                    transcript_text = transcript_data["text"]
                    transcript_path = self.run_dir / "transcript.txt"
                    transcript_path.write_text(transcript_text, encoding="utf-8")

        # Calculate and save metrics
        self._calculate_metrics(total_time)
        self._write_json("metrics.json", self.metrics)

        _logger.info(f"💾 Experiment saved: {self.run_dir}")

    def _calculate_metrics(self, total_time: float):
        """Calculate automatic metrics."""
        # Timing metrics
        self.metrics["timing"] = {
            "total_seconds": total_time,
            **self.timing_data,
        }

        # LLM call metrics
        if self.pydantic_calls:
            total_cost = sum(call.get("cost_usd", 0) for call in self.pydantic_calls)
            total_latency = sum(
                call.get("latency_seconds", 0) for call in self.pydantic_calls
            )
            self.metrics["llm"] = {
                "total_calls": len(self.pydantic_calls),
                "total_cost_usd": total_cost,
                "total_latency_seconds": total_latency,
            }

        # Count metrics from outputs
        if "extracted_claims" in self.outputs:
            claims_data = self.outputs["extracted_claims"]
            if isinstance(claims_data, dict):
                self.metrics["claims_extracted"] = claims_data.get("total_count", 0)

        # Success indicator
        self.metrics["success"] = True
        self.metrics["error"] = None

    def mark_error(self, error: Exception):
        """Mark experiment as failed."""
        self.metrics["success"] = False
        self.metrics["error"] = str(error)
        _logger.error(f"❌ Experiment failed: {error}")

    def _write_json(self, filename: str, data: Any):
        """Write data to JSON file."""
        filepath = self.run_dir / filename
        with filepath.open("w") as f:
            json.dump(data, f, indent=2, default=str)

    def __enter__(self):
        """Context manager entry."""
        ExperimentTracker.set_current(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is not None:
            self.mark_error(exc_val)
        self.save()
        ExperimentTracker.set_current(None)


@contextmanager
def timer(label: str):
    """Time a code block and log to current tracker if available."""
    start = time.time()
    try:
        yield
    finally:
        duration = time.time() - start
        _logger.info(f"⏱️  {label}: {duration:.2f}s")

        # Auto-track if tracker is active
        tracker = ExperimentTracker.get_current()
        if tracker:
            tracker.log_timing(label, duration)
