import inspect
import logging
import time
from contextvars import ContextVar
from datetime import datetime
from functools import wraps
from typing import Any, Awaitable, Callable, TypeVar, cast

from pydantic_ai import Agent

from factible.evaluation.tracker import ExperimentTracker
from factible.models.llm import get_model_pricing

_logger = logging.getLogger(__name__)

T = TypeVar("T")

_current_component: ContextVar[str | None] = ContextVar(
    "current_pydantic_component", default=None
)

_original_agent_run = Agent.run
_original_agent_run_sync = Agent.run_sync
_agent_patch_initialized = False


def _ensure_agent_patched() -> None:
    global _agent_patch_initialized
    if _agent_patch_initialized:
        return
    _agent_patch_initialized = True

    async def _tracked_agent_run(self, prompt, *run_args, **run_kwargs):  # type: ignore[override]
        tracker = ExperimentTracker.get_current()
        component = _current_component.get()
        start_time = time.time()
        result = await _original_agent_run(self, prompt, *run_args, **run_kwargs)
        if not tracker or not component:
            return result

        latency = time.time() - start_time
        output = result.output if hasattr(result, "output") else result
        input_tokens = estimate_tokens(prompt)
        output_str = (
            str(output)
            if not hasattr(output, "model_dump_json")
            else output.model_dump_json()
        )
        output_tokens = estimate_tokens(output_str)
        cost = calculate_cost(str(self.model), input_tokens, output_tokens)

        call_data = {
            "component": component,
            "model": str(self.model),
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

    def _tracked_agent_run_sync(self, prompt, *run_args, **run_kwargs):  # type: ignore[override]
        tracker = ExperimentTracker.get_current()
        component = _current_component.get()
        start_time = time.time()
        result = _original_agent_run_sync(self, prompt, *run_args, **run_kwargs)
        if not tracker or not component:
            return result

        latency = time.time() - start_time
        output = result.output if hasattr(result, "output") else result
        input_tokens = estimate_tokens(prompt)
        output_str = (
            str(output)
            if not hasattr(output, "model_dump_json")
            else output.model_dump_json()
        )
        output_tokens = estimate_tokens(output_str)
        cost = calculate_cost(str(self.model), input_tokens, output_tokens)

        call_data = {
            "component": component,
            "model": str(self.model),
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

    Agent.run = _tracked_agent_run
    Agent.run_sync = _tracked_agent_run_sync


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
    _ensure_agent_patched()
    token = _current_component.set(component)
    try:
        agent_method = getattr(agent, method)
        return agent_method(prompt)
    finally:
        _current_component.reset(token)


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
        is_async = inspect.iscoroutinefunction(func)
        _ensure_agent_patched()

        if is_async:
            async_func = cast(Callable[..., Awaitable[T]], func)

            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> T:
                token = _current_component.set(component)
                try:
                    return await async_func(*args, **kwargs)
                finally:
                    _current_component.reset(token)

            return async_wrapper  # type: ignore[return-value]
        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs) -> T:
                token = _current_component.set(component)
                try:
                    return func(*args, **kwargs)
                finally:
                    _current_component.reset(token)

            return sync_wrapper

    return decorator
