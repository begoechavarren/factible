<div align="center">
  <h1>🕵️‍♂️ factible</h1>
  <p><em>E2E project to fact check YouTube videos</em></p>
</div>

```
$ cd backend
$ uv run python -m factible_api.main

$ ollama serve

$ cd frontend
$ npm run dev

$ ./factible/experiments/run_phase1_experiments.sh --ofat-only
$ ./factible/experiments/run_phase1_experiments.sh --resume vary_claims_20251206_191228 --ofat-only
$ uv run factible-experiments analyze --runs-dir factible/experiments/runs/vary_claims_20251206_191228 --name vary_claims_partial

$ uv run factible-experiments run --experiment vary_claims --runs-subdir vary_claims_20251206_191228
```
