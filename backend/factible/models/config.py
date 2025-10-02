from factible.models.llm import ModelChoice

# TODO: Add model config to components / steps instead of centralized?
CLAIM_EXTRACTOR_MODEL = ModelChoice.OLLAMA_QWEN3_0_1_7B
QUERY_GENERATOR_MODEL = ModelChoice.OLLAMA_QWEN3_0_1_7B
EVIDENCE_EXTRACTOR_MODEL = ModelChoice.OLLAMA_QWEN3_0_1_7B
OUTPUT_GENERATOR_MODEL = ModelChoice.OLLAMA_QWEN3_0_1_7B
