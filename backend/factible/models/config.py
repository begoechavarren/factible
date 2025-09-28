from factible.models.llm import ModelChoice

# TODO: Add model config to components / steps instead of centralized?
CLAIM_EXTRACTOR_MODEL = ModelChoice.OLLAMA_QWEN3_0_8B
QUERY_GENERATOR_MODEL = ModelChoice.OLLAMA_QWEN3_0_8B
EVIDENCE_EXTRACTOR_MODEL = ModelChoice.OLLAMA_QWEN3_0_8B
OUTPUT_GENERATOR_MODEL = ModelChoice.OLLAMA_QWEN3_0_8B
