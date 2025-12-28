import logging
from typing import Generic, TypeVar
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models import ModelSettings

from factible.models.llm import get_model, ModelChoice

_logger = logging.getLogger(__name__)

# Type variable for response models
T = TypeVar("T", bound=BaseModel)


class LLMJudgeBase(Generic[T]):
    """
    Base class for LLM-as-judge evaluations.

    Provides a simple, robust pattern for using LLM to evaluate different aspects
    of the fact-checking system.
    """

    def __init__(
        self,
        response_model: type[T],
        system_prompt: str,
        model_choice: ModelChoice = ModelChoice.OLLAMA_QWEN3_0_8B,
        temperature: float = 0.0,
        max_tokens: int = 500,
    ):
        """
        Initialize LLM judge.

        Args:
            response_model: Pydantic model for structured responses
            system_prompt: System prompt defining the evaluation task
            model_choice: Which LLM to use (default: Ollama Qwen3 8B to avoid
                circularity with the pipeline's default GPT-4o-mini)
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Maximum response tokens
        """
        self.response_model = response_model
        self.system_prompt = system_prompt

        # Create pydantic-ai agent
        self.agent = Agent(
            model=get_model(model_choice),
            output_type=response_model,  # type: ignore[arg-type]
            system_prompt=system_prompt,
        )

        self.model_settings: ModelSettings = {
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

    async def evaluate_async(self, prompt: str, **kwargs) -> T:
        """
        Run LLM evaluation asynchronously.

        Args:
            prompt: User prompt with content to evaluate
            **kwargs: Additional arguments passed to agent.run()

        Returns:
            Instance of response_model with evaluation results
        """
        try:
            result = await self.agent.run(
                prompt, model_settings=self.model_settings, **kwargs
            )
            return result.output
        except Exception as e:
            _logger.error(f"LLM judge evaluation failed: {e}")
            raise

    def evaluate_sync(self, prompt: str, **kwargs) -> T:
        """
        Run LLM evaluation synchronously.

        Args:
            prompt: User prompt with content to evaluate
            **kwargs: Additional arguments passed to agent.run_sync()

        Returns:
            Instance of response_model with evaluation results
        """
        try:
            result = self.agent.run_sync(
                prompt, model_settings=self.model_settings, **kwargs
            )
            return result.output
        except Exception as e:
            _logger.error(f"LLM judge evaluation failed: {e}")
            raise


def create_simple_judge(
    response_model: type[T],
    system_prompt: str,
    **kwargs,
) -> LLMJudgeBase[T]:
    """
    Factory function to create a simple LLM judge.

    Args:
        response_model: Pydantic model for responses
        system_prompt: Evaluation instructions
        **kwargs: Additional arguments for LLMJudgeBase

    Returns:
        Configured LLMJudgeBase instance
    """
    return LLMJudgeBase(
        response_model=response_model,
        system_prompt=system_prompt,
        **kwargs,
    )
