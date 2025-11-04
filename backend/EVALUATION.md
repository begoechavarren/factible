# Evaluation Framework

## Overview

The evaluation framework automatically tracks all fact-checking experiments with structured metrics, timing data, and full input/output logging. Every run creates a timestamped directory with JSON files for easy analysis.

## Structure

```
experiments/
└── runs/
    └── {timestamp}_{component}_{experiment_name}/
        ├── config.json              # Experiment configuration
        ├── timing.json              # Execution timing per step
        ├── pydantic_calls.json      # All LLM calls with inputs/outputs
        ├── inputs.json              # Pipeline inputs
        ├── outputs.json             # Pipeline outputs
        └── metrics.json             # Calculated metrics
```

## What's Tracked Automatically

### 1. Timing Metrics
- Total pipeline execution time
- Per-component timing (transcript extraction, claim extraction, etc.)
- Individual LLM call latency

### 2. Cost Metrics
- Estimated token usage (input/output)
- Estimated cost in USD per LLM call
- Total pipeline cost

### 3. Pydantic AI Calls
Each LLM call logs:
- Component name (claim_extraction, query_generation, etc.)
- Model used (gpt-4o-mini, qwen3:8b, etc.)
- Input prompt (full text)
- Output (structured data)
- Latency in seconds
- Estimated tokens and cost

### 4. Inputs/Outputs
- Full transcript text
- Extracted claims (all fields)
- Generated queries
- Search results (processed, not raw HTML)
- Final verdicts

## Usage

### Running with Tracking (Command Line)

```python
from factible.run import run_factible

result = run_factible(
    video_url="https://youtube.com/watch?v=...",
    experiment_name="baseline_gpt4o",  # Name your experiment
    max_claims=5,
    max_queries_per_claim=2,
    max_results_per_query=3
)
```

### Running via API

```bash
curl -X POST http://localhost:8000/api/v1/fact-check/stream \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://youtube.com/watch?v=...",
    "experiment_name": "test_run",
    "max_claims": 5,
    "max_queries_per_claim": 2,
    "max_results_per_query": 3
  }'
```

## Experiment Output Files

### config.json
```json
{
  "run_id": "20251201_210634_end_to_end_baseline",
  "experiment_name": "baseline",
  "component": "end_to_end",
  "timestamp": "2025-12-01T21:06:34Z",
  "video_url": "https://youtube.com/watch?v=...",
  "max_claims": 5,
  "max_queries_per_claim": 2,
  "max_results_per_query": 3,
  "headless_search": true
}
```

### timing.json
```json
{
  "Step 1: Transcript extraction": 0.90,
  "Step 2: Claim extraction": 15.09,
  "  Query generation for claim 1": 7.40,
  "    Search execution for query 1": 9.81,
  "Step 4: Output generation": 2.47,
  "total_seconds": 35.68
}
```

### pydantic_calls.json
```json
[
  {
    "component": "claim_extraction",
    "model": "openai:gpt-4o-mini",
    "timestamp": "2025-12-01T21:06:48Z",
    "latency_seconds": 14.09,
    "input_prompt": "Extract all factual claims from this YouTube transcript...",
    "input_length_chars": 2847,
    "input_tokens_estimated": 711,
    "output": {
      "claims": [...],
      "total_count": 12
    },
    "output_length_chars": 1523,
    "output_tokens_estimated": 380,
    "cost_usd": 0.000334
  }
]
```

### metrics.json
```json
{
  "timing": {
    "total_seconds": 35.68,
    "Step 1: Transcript extraction": 0.90,
    "Step 2: Claim extraction": 15.09
  },
  "pydantic_ai": {
    "total_calls": 4,
    "total_cost_usd": 0.0124,
    "total_latency_seconds": 29.63
  },
  "claims_extracted": 12,
  "success": true,
  "error": null
}
```

## Comparing Experiments

### Manual Comparison
Simply open the JSON files from different experiments and compare:

```bash
# View timing for experiment 1
cat experiments/runs/20251201_210634_end_to_end_baseline/timing.json

# View timing for experiment 2
cat experiments/runs/20251201_215423_end_to_end_qwen/timing.json
```

### Programmatic Analysis (Future Phase 2)
```python
# Coming in Phase 2: Analysis tools
from factible.evaluation.analysis import ExperimentComparison

comp = ExperimentComparison()
df = comp.load_experiments("end_to_end")
print(df[["experiment_name", "total_cost_usd", "total_seconds"]])
```

## Model Configuration

The framework automatically detects which model is used for each component from `factible/models/config.py`:

```python
CLAIM_EXTRACTOR_MODEL = ModelChoice.OPENAI_GPT4O_MINI
QUERY_GENERATOR_MODEL = ModelChoice.OPENAI_GPT4O_MINI
EVIDENCE_EXTRACTOR_MODEL = ModelChoice.OPENAI_GPT4O_MINI
OUTPUT_GENERATOR_MODEL = ModelChoice.OPENAI_GPT4O_MINI
```

## Tips for Thesis Evaluation

### Experiment Naming Convention
Use descriptive names that encode the configuration:
- `baseline_gpt4o_mini` - Baseline with GPT-4o-mini
- `qwen3_8b_v1` - Qwen3 8B model, version 1
- `gpt4o_mini_max10claims` - GPT-4o-mini with 10 claims
- `claude_sonnet_queries3` - Claude Sonnet with 3 queries per claim

### Organizing Experiments
```
experiments/runs/
├── 20251201_baseline_gpt4o_mini/
├── 20251201_baseline_qwen3_8b/
├── 20251202_optimized_gpt4o_mini/
├── 20251202_optimized_claude_sonnet/
└── ...
```

### What to Compare
- **Cost vs Quality**: Compare `total_cost_usd` across models
- **Latency vs Quality**: Compare `total_seconds` across models
- **Component Performance**: Compare individual step timing
- **Token Efficiency**: Compare tokens used per component

## Next Steps (Phase 2)

1. **Ground Truth Dataset**: Create test videos with expected outputs
2. **Component Metrics**: Precision/Recall for claim extraction, quality scores for queries
3. **Analysis Tools**: Scripts to compare experiments and generate thesis tables/plots
4. **LLM-as-Judge**: Automated quality evaluation for outputs

## Troubleshooting

### No experiments directory created
- Check that you're running from the correct directory
- The tracker creates `experiments/runs/` relative to working directory

### Missing pydantic_calls.json
- Ensure decorators are applied to component functions
- Check that Pydantic AI calls are being made

### Incomplete outputs.json
- Check for errors during execution
- Review `metrics.json` for `success: false` and `error` field
