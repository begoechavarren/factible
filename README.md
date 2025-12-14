<div align="center">
  <h1>🕵️‍♂️ factible</h1>
  <p><em>E2E project to fact check YouTube videos</em></p>
</div>

<p align="center">
  <img src="assets/main.png" alt="Landing page" /><br/>
  <em>Landing page</em>
</p>

<p align="center">
  <img src="assets/process.png" alt="Processing" /><br/>
  <em>Real-time updates</em>
</p>

<p align="center">
  <img src="assets/results.png" alt="Results overview" /><br/>
  <em>Results page</em>
</p>

<p align="center">
  <img src="assets/claim.png" alt="Claim details" /><br/>
  <em>Results per claim</em>
</p>

## Quick Start

```bash
# Backend
cd backend
uv run python -m factible_api.main

# Frontend
cd frontend
npm run dev
```

```bash
# Run experiments
uv run factible-experiments run --experiment vary_claims

# Analyze results
uv run factible-experiments analyze --runs-dir factible/experiments/runs/<run_dir>
```
