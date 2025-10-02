from enum import Enum
from functools import lru_cache
from typing import Union

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider


class ModelChoice(Enum):
    OPENAI_GPT4O_MINI = ("openai", "gpt-4o-mini")  # 128K context window
    # https://ollama.com/library/qwen3
    OLLAMA_QWEN3_0_8B = ("ollama", "qwen3:8b")  # 40K context window
    OLLAMA_QWEN3_0_4B = ("ollama", "qwen3:4b")  # # 256K context window
    OLLAMA_QWEN3_0_1_7B = ("ollama", "qwen3:1.7b")  # 40K context window

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
