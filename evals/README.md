# Evaluations

Phase 7 implements the credential-free CrossScopeMem benchmark and a reproducible evaluation harness. External benchmark data is never committed or fabricated; LongMemEval, LoCoMo, and MemConflict remain Phase 8 adapters.

Run all four controlled backends:

```bash
make eval-all
```

Run one backend or an ablation:

```bash
PYTHONPATH=src uv run python -m evals.runners.run_eval \
  --dataset cross_scope_mem --system scopegraph --config configs/experiments.yaml

PYTHONPATH=src uv run python -m evals.runners.run_ablation \
  --dataset cross_scope_mem --system scopegraph --ablation no_graph_traversal
```

Generate processed JSON, Markdown, and SVG output from raw JSONL:

```bash
PYTHONPATH=src uv run python -m evals.analysis.run_report results/raw/*.jsonl
```

The raw record preserves the scenario, question, gold state, retrieved IDs/scopes/scores, trace, latency, token count, storage statistics, configuration hash, seed, and git commit. Scoring is a separate pass. The default answer field uses a transparent deterministic fact extractor; no LLM answer is claimed.

The correction-persistence experiment compares no correction, conversational correction, and direct graph correction at +1, +5, +10, and +20 sessions:

```bash
PYTHONPATH=src uv run python -m evals.runners.run_correction_eval
```
