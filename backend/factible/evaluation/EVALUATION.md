# Evaluation Framework

## Overview

The evaluation framework automatically tracks all fact-checking experiments with structured metrics, timing data, and full input/output logging. Every run creates a timestamped directory with JSON files for easy analysis.

## Quick Start - Automated Experiments

### 1. Configure Your Test Videos

Edit `scripts/experiments/experiments_config.yaml` to define your test videos and experiment configurations:

```yaml
videos:
  - id: "video_1"
    url: "https://youtube.com/watch?v=..."
    description: "Political speech with verifiable claims"
    tags: ["politics", "baseline"]

experiments:
  - name: "baseline_gpt4o_mini"
    description: "Baseline configuration"
    model_config: "OPENAI_GPT4O_MINI"
    max_claims: 5
    max_queries_per_claim: 2
    max_results_per_query: 3
```

### 2. Run Experiments

```bash
# Run all experiments on all videos
python scripts/experiments/run_experiments.py

# Run specific experiment
python scripts/experiments/run_experiments.py --experiment baseline_gpt4o_mini

# Run on specific video
python scripts/experiments/run_experiments.py --video video_1

# Dry run (preview what will execute)
python scripts/experiments/run_experiments.py --dry-run
```

### 3. Analyze Results

Results are automatically saved to `experiments/runs/{timestamp}_{experiment}_{video}/`

Each experiment creates:
- `config.json` - Experiment parameters
- `timing.json` - Execution duration per step
- `pydantic_calls.json` - All LLM calls with inputs/outputs
- `inputs.json` - Pipeline inputs
- `outputs.json` - Structured results
- `metrics.json` - Aggregated metrics

## Manual Experiments

You can also run experiments programmatically:

```python
from factible.run import run_factible

result = run_factible(
    video_url="https://youtube.com/watch?v=...",
    experiment_name="my_test_run",
    max_claims=5,
    max_queries_per_claim=2,
    max_results_per_query=3
)
```

Or via the API:

```bash
curl -X POST http://localhost:8000/api/v1/fact-check/stream \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://youtube.com/watch?v=...",
    "experiment_name": "test_run",
    "max_claims": 5
  }'
```

## Experiment Configuration Reference

### Video Configuration

```yaml
videos:
  - id: "unique_identifier"              # Required: Used in experiment naming
    url: "https://youtube.com/watch?v=..." # Required: YouTube video URL
    description: "Human readable desc"     # Optional: For documentation
    tags: ["category", "type"]             # Optional: For filtering experiments
    notes: "Additional context"            # Optional: For your reference
```

### Experiment Configuration

```yaml
experiments:
  - name: "experiment_name"              # Required: Unique identifier
    description: "What this tests"       # Optional: For documentation
    model_config: "OPENAI_GPT4O_MINI"   # Required: ModelChoice enum name
    max_claims: 5                        # Optional: Default 5
    max_queries_per_claim: 2            # Optional: Default 2
    max_results_per_query: 3            # Optional: Default 3
    enable_search: true                  # Optional: Default true
    video_filter: []                     # Optional: Empty = all videos, or list of tags
```

#### Available Model Configs

- `OPENAI_GPT4O_MINI` - GPT-4o Mini (fast, cheap, 128K context)
- `OPENAI_GPT4O` - GPT-4o (high quality, expensive, 128K context)
- `OPENAI_GPT4_TURBO` - GPT-4 Turbo (high quality, very expensive, 128K context)
- `OLLAMA_QWEN3_0_8B` - Qwen3 8B (local, free, 40K context)
- `OLLAMA_QWEN3_0_4B` - Qwen3 4B (local, free, 256K context)
- `OLLAMA_QWEN3_0_1_7B` - Qwen3 1.7B (local, free, 40K context)

**Note:** The model_config in experiments_config.yaml sets which models to test. The actual models used by each component are configured in `factible/models/config.py`.

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

## Experiment Output Files

### config.json
```json
{
  "run_id": "20251201_210634_end_to_end_baseline_video1",
  "experiment_name": "baseline_video1",
  "component": "end_to_end",
  "timestamp": "2025-12-01T21:06:34Z",
  "video_url": "https://youtube.com/watch?v=...",
  "max_claims": 5,
  "max_queries_per_claim": 2,
  "max_results_per_query": 3
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

### metrics.json
```json
{
  "timing": {
    "total_seconds": 35.68
  },
  "pydantic_ai": {
    "total_calls": 4,
    "total_cost_usd": 0.0124,
    "total_latency_seconds": 29.63
  },
  "claims_extracted": 12,
  "success": true
}
```

## Tips for Thesis Evaluation

### Experiment Naming Convention
Use descriptive names that encode the configuration:
- `baseline_gpt4o_mini_video1` - Baseline with GPT-4o-mini on video 1
- `fast_qwen3_4b_video2` - Qwen3 4B model on video 2
- `high_quality_gpt4o_video1` - GPT-4o with max quality settings

The experiment runner automatically appends `_{video_id}` to experiment names.

### Organizing Test Videos
Group videos by category using tags:
```yaml
videos:
  - id: "politics_1"
    tags: ["politics", "baseline"]
  - id: "science_1"
    tags: ["science", "complex"]
  - id: "health_1"
    tags: ["health", "baseline"]
```

Then filter experiments:
```yaml
experiments:
  - name: "baseline"
    video_filter: ["baseline"]  # Only baseline-tagged videos
  - name: "full_test"
    video_filter: []  # All videos
```

### What to Compare
- **Cost vs Quality**: Compare `total_cost_usd` across models
- **Latency vs Quality**: Compare `total_seconds` across models
- **Component Performance**: Compare individual step timing
- **Token Efficiency**: Compare tokens used per component

### Analyzing Results

Compare experiments manually:
```bash
# View metrics from different runs
cat experiments/runs/20251201_*_baseline_video1/metrics.json | jq '.pydantic_ai.total_cost_usd'
cat experiments/runs/20251201_*_fast_qwen_video1/metrics.json | jq '.pydantic_ai.total_cost_usd'
```

Or write analysis scripts using the structured JSON data:
```python
import json
from pathlib import Path

# Load all experiment results
results = []
for run_dir in Path("experiments/runs").iterdir():
    if run_dir.is_dir():
        with open(run_dir / "metrics.json") as f:
            metrics = json.load(f)
        with open(run_dir / "config.json") as f:
            config = json.load(f)
        results.append({"config": config, "metrics": metrics})

# Analyze
import pandas as pd
df = pd.DataFrame(results)
print(df[["config.experiment_name", "metrics.pydantic_ai.total_cost_usd"]])
```

## Troubleshooting

### No experiments directory created
- Check that you're running from the backend directory
- The tracker creates `experiments/runs/` relative to working directory

### Missing pydantic_calls.json
- Ensure decorators are applied to component functions
- Check that Pydantic AI calls are being made

### Experiment runner fails
- Verify your `.env` file has required API keys (OPENAI_API_KEY)
- Check that video URLs are valid and have transcripts
- Run with `--dry-run` first to validate configuration

### Model not found error
- Ensure model_config in experiments_config.yaml matches a ModelChoice enum value
- For local models (Ollama), ensure Ollama is running: `ollama serve`

## Next Steps (Future Phases)

1. **Ground Truth Dataset**: Create test videos with expected outputs
2. **Component Metrics**: Precision/Recall for claim extraction, quality scores for queries
3. **Analysis Tools**: Scripts to compare experiments and generate thesis tables/plots
4. **LLM-as-Judge**: Automated quality evaluation for outputs
