from enum import Enum
from functools import lru_cache
from typing import Union

from pydantic import BaseModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider


class ModelConfig(BaseModel):
    """Configuration for a language model including provider and pricing."""

    provider: str
    model_name: str
    price_input_per_million: float
    price_output_per_million: float
    context_window: int


class ModelChoice(Enum):
    """Available language models with their configurations."""

    OPENAI_GPT4O_MINI = ModelConfig(
        provider="openai",
        model_name="gpt-4o-mini",
        price_input_per_million=0.150,
        price_output_per_million=0.600,
        context_window=128_000,
    )
    OPENAI_GPT4O = ModelConfig(
        provider="openai",
        model_name="gpt-4o",
        price_input_per_million=5.00,
        price_output_per_million=15.00,
        context_window=128_000,
    )
    OPENAI_GPT4_TURBO = ModelConfig(
        provider="openai",
        model_name="gpt-4-turbo",
        price_input_per_million=10.00,
        price_output_per_million=30.00,
        context_window=128_000,
    )
    # https://ollama.com/library/qwen3 - Local models, no cost
    OLLAMA_QWEN3_0_8B = ModelConfig(
        provider="ollama",
        model_name="qwen3:8b",
        price_input_per_million=0.0,
        price_output_per_million=0.0,
        context_window=40_000,
    )
    OLLAMA_QWEN3_0_4B = ModelConfig(
        provider="ollama",
        model_name="qwen3:4b",
        price_input_per_million=0.0,
        price_output_per_million=0.0,
        context_window=256_000,
    )
    OLLAMA_QWEN3_0_1_7B = ModelConfig(
        provider="ollama",
        model_name="qwen3:1.7b",
        price_input_per_million=0.0,
        price_output_per_million=0.0,
        context_window=40_000,
    )


ModelSpec = Union[str, OpenAIChatModel]

_OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"


def get_model_pricing(model_str: str) -> tuple[float, float]:
    """
    Get pricing for a model string (e.g., 'openai:gpt-4o-mini').

    Returns:
        (price_input_per_million, price_output_per_million)
    """
    # Extract base model name from string like "openai:gpt-4o-mini"
    model_name = model_str.split(":")[-1]

    # Search ModelChoice enum for matching model
    for choice in ModelChoice:
        if choice.value.model_name == model_name:
            return (
                choice.value.price_input_per_million,
                choice.value.price_output_per_million,
            )

    # Default to 0.0 if not found (e.g., unknown models)
    return (0.0, 0.0)


@lru_cache(maxsize=None)
def _get_ollama_model(model_name: str) -> OpenAIChatModel:
    provider = OllamaProvider(base_url=_OLLAMA_BASE_URL)
    return OpenAIChatModel(model_name=model_name, provider=provider)


def get_model(choice: ModelChoice) -> ModelSpec:
    """Get a model instance from a ModelChoice enum value."""
    config = choice.value
    if config.provider == "openai":
        return f"openai:{config.model_name}"
    if config.provider == "ollama":
        return _get_ollama_model(config.model_name)
    raise ValueError(f"Unsupported model provider: {config.provider}")
