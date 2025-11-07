# Factible Scripts

Utility scripts for running and analyzing experiments.

## experiments/

Contains experiment runner and configuration files.

### run_experiments.py

Automated experiment runner that executes fact-checking experiments based on YAML configuration.

**Usage:**

```bash
# Run all experiments on all videos
python scripts/experiments/run_experiments.py

# Run specific experiment
python scripts/experiments/run_experiments.py --experiment baseline_gpt4o_mini

# Run on specific video
python scripts/experiments/run_experiments.py --video example_video_1

# Preview what will be executed (dry run)
python scripts/experiments/run_experiments.py --dry-run

# Custom config file
python scripts/experiments/run_experiments.py --config my_experiments.yaml

# Change log level
python scripts/experiments/run_experiments.py --log-level DEBUG
```

### experiments_config.yaml

Configuration file defining:
- Test videos with URLs, descriptions, and tags
- Experiment configurations with model choices and parameters
- Global settings

See `factible/evaluation/EVALUATION.md` for detailed configuration reference.

### Output

Results are saved to `experiments/runs/{timestamp}_{experiment}_{video}/` with:
- `config.json` - Experiment parameters
- `timing.json` - Execution timing
- `pydantic_calls.json` - LLM calls with inputs/outputs
- `metrics.json` - Aggregated metrics
- `inputs.json` - Pipeline inputs
- `outputs.json` - Structured results

### Examples

```bash
# Run all baseline experiments
python scripts/experiments/run_experiments.py --experiment baseline_gpt4o_mini

# Test configuration without executing
python scripts/experiments/run_experiments.py --dry-run

# Run experiments on a specific test video
python scripts/experiments/run_experiments.py --video politics_claim_1
```

## Future Scripts

- `analyze_experiments.py` - Compare experiment results and generate tables
- `validate_ground_truth.py` - Validate experiments against ground truth dataset
- `export_results.py` - Export results to CSV/Excel for thesis analysis
