from enum import Enum
from functools import lru_cache
from typing import Union

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider


class ModelChoice(Enum):
    OPENAI_GPT4O_MINI = ("openai", "gpt-4o-mini")
    OLLAMA_QWEN3_0_8B = ("ollama", "qwen3:8b")
    OLLAMA_QWEN3_0_0_5B = ("ollama", "qwen3:0.6b")

    def __init__(self, provider: str, model_name: str) -> None:
        self.provider = provider
        self.model_name = model_name


ModelSpec = Union[str, OpenAIChatModel]

_OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"


@lru_cache(maxsize=None)
def _get_ollama_model(model_name: str) -> OpenAIChatModel:
    provider = OllamaProvider(base_url=_OLLAMA_BASE_URL)
    return OpenAIChatModel(model_name=model_name, provider=provider)


def get_model(choice: ModelChoice) -> ModelSpec:
    if choice.provider == "openai":
        return f"openai:{choice.model_name}"
    if choice.provider == "ollama":
        return _get_ollama_model(choice.model_name)
    raise ValueError(f"Unsupported model provider: {choice.provider}")
