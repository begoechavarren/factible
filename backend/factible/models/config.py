from factible.models.llm import ModelChoice

# TODO: Add model config to components / steps instead of centralized?
CLAIM_EXTRACTOR_MODEL = ModelChoice.OPENAI_GPT4O_MINI
QUERY_GENERATOR_MODEL = ModelChoice.OPENAI_GPT4O_MINI
EVIDENCE_EXTRACTOR_MODEL = ModelChoice.OPENAI_GPT4O_MINI
OUTPUT_GENERATOR_MODEL = ModelChoice.OPENAI_GPT4O_MINI
