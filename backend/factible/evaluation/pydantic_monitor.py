import logging
import time
from datetime import datetime
from functools import wraps
from typing import Any, Callable, TypeVar

from pydantic_ai import Agent

from factible.evaluation.tracker import ExperimentTracker
from factible.models.llm import get_model_pricing

_logger = logging.getLogger(__name__)

T = TypeVar("T")


def estimate_tokens(text: str) -> int:
    """Rough token estimation (1 token ≈ 4 characters for English)."""
    return len(text) // 4


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD for a model call."""
    price_input, price_output = get_model_pricing(model)
    input_cost = (input_tokens / 1_000_000) * price_input
    output_cost = (output_tokens / 1_000_000) * price_output
    return input_cost + output_cost


def track_pydantic_call(
    agent: Agent,
    prompt: str,
    component: str,
    method: str = "run_sync",
) -> Any:
    """
    Wrapper for Pydantic AI agent calls that tracks metrics.

    Args:
        agent: The Pydantic AI agent
        prompt: The prompt to send
        component: Component name (e.g., 'claim_extraction')
        method: Agent method to call ('run_sync' or 'run')

    Returns:
        The agent result
    """
    tracker = ExperimentTracker.get_current()

    # Get model name
    model_name = str(agent.model) if hasattr(agent, "model") else "unknown"

    # Timing
    start_time = time.time()

    # Execute agent call
    agent_method = getattr(agent, method)
    result = agent_method(prompt)

    latency = time.time() - start_time

    # Extract output
    output = result.output if hasattr(result, "output") else result

    # Estimate tokens and cost
    input_tokens = estimate_tokens(prompt)
    output_str = (
        str(output)
        if not hasattr(output, "model_dump_json")
        else output.model_dump_json()
    )
    output_tokens = estimate_tokens(output_str)
    cost = calculate_cost(model_name, input_tokens, output_tokens)

    # Log to tracker
    if tracker:
        call_data = {
            "component": component,
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "latency_seconds": round(latency, 2),
            "input_prompt": prompt,
            "input_length_chars": len(prompt),
            "input_tokens_estimated": input_tokens,
            "output": output.model_dump()
            if hasattr(output, "model_dump")
            else str(output),
            "output_length_chars": len(output_str),
            "output_tokens_estimated": output_tokens,
            "cost_usd": round(cost, 6),
        }
        tracker.log_pydantic_call(call_data)

    return result


def track_pydantic(component: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to automatically track Pydantic AI calls in a function.

    Usage:
        @track_pydantic("claim_extraction")
        def extract_claims(transcript: str) -> ExtractedClaims:
            agent = _get_claim_extractor_agent()
            result = agent.run_sync(prompt)
            return result.output

    Args:
        component: Component name for tracking
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Monkey-patch agent.run_sync to track calls
            original_run_sync = Agent.run_sync

            def tracked_run_sync(self, prompt, *run_args, **run_kwargs):
                return track_pydantic_call(self, prompt, component, "run_sync").output

            try:
                Agent.run_sync = tracked_run_sync
                return func(*args, **kwargs)
            finally:
                Agent.run_sync = original_run_sync

        return wrapper

    return decorator
