import inspect
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
            "output": output.model_dump(exclude_none=True)
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
    Supports both sync and async functions.

    Usage:
        @track_pydantic("claim_extraction")
        def extract_claims(transcript: str) -> ExtractedClaims:
            agent = _get_claim_extractor_agent()
            result = agent.run_sync(prompt)
            return result.output

        @track_pydantic("evidence_extraction")
        async def extract_evidence(query: str, content: str) -> Evidence:
            agent = _get_evidence_agent()
            result = await agent.run(prompt)
            return result.output

    Args:
        component: Component name for tracking
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Check if function is async
        is_async = inspect.iscoroutinefunction(func)

        if is_async:

            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> T:
                # Save original async method before patching
                original_run = Agent.run

                async def tracked_run(self, prompt, *run_args, **run_kwargs):
                    tracker = ExperimentTracker.get_current()
                    model_name = (
                        str(self.model) if hasattr(self, "model") else "unknown"
                    )

                    start_time = time.time()

                    # Call the original async method
                    result = await original_run(self, prompt, *run_args, **run_kwargs)

                    latency = time.time() - start_time
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
                            "output": output.model_dump(exclude_none=True)
                            if hasattr(output, "model_dump")
                            else str(output),
                            "output_length_chars": len(output_str),
                            "output_tokens_estimated": output_tokens,
                            "cost_usd": round(cost, 6),
                        }
                        tracker.log_pydantic_call(call_data)

                    return result

                try:
                    Agent.run = tracked_run
                    return await func(*args, **kwargs)  # type: ignore[misc]
                finally:
                    Agent.run = original_run

            return async_wrapper  # type: ignore[return-value]
        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs) -> T:
                # Save original sync method before patching
                original_run_sync = Agent.run_sync

                def tracked_run_sync(self, prompt, *run_args, **run_kwargs):
                    tracker = ExperimentTracker.get_current()
                    model_name = (
                        str(self.model) if hasattr(self, "model") else "unknown"
                    )

                    start_time = time.time()

                    # Call the original method, not the patched one
                    result = original_run_sync(self, prompt, *run_args, **run_kwargs)

                    latency = time.time() - start_time
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
                            "output": output.model_dump(exclude_none=True)
                            if hasattr(output, "model_dump")
                            else str(output),
                            "output_length_chars": len(output_str),
                            "output_tokens_estimated": output_tokens,
                            "cost_usd": round(cost, 6),
                        }
                        tracker.log_pydantic_call(call_data)

                    return result

                try:
                    Agent.run_sync = tracked_run_sync
                    return func(*args, **kwargs)
                finally:
                    Agent.run_sync = original_run_sync

            return sync_wrapper  # type: ignore[return-value]

    return decorator
