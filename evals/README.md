# Evaluations

Phase 7 implements the credential-free CrossScopeMem benchmark and a reproducible evaluation harness. External local-file adapters cover LongMemEval, LongMemEval-V2, LoCoMo, MemConflict, MemoryAgentBench, RHELM, MemBench, Mem2ActBench, and TIME; benchmark data is never committed or fabricated.

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

For end-to-end answer evaluation with the configured provider, add `LIVE_ANSWER=1`:

```bash
make eval-all LIVE_ANSWER=1
```

Use `--scenario-count` with the module runner to control the number of generated histories. Keep the default deterministic mode for retrieval-only comparisons and use the same live answer model across every architecture when comparing answer accuracy.

For the complete live pipeline, use real extraction, cached OpenAI embeddings, and the configured answer model together:

```bash
make eval-all LIVE=1
```

Reports also write paired bootstrap intervals to `results/processed/confidence_intervals.json`, comparing each system with ScopeGraph by scenario and question.

The correction-persistence experiment compares no correction, conversational correction, and direct graph correction at +1, +5, +10, and +20 sessions:

```bash
PYTHONPATH=src uv run python -m evals.runners.run_correction_eval
```

Validate an acquired external release before replay:

```bash
make validate-external DATASET=longmemeval DATA_PATH=data/longmemeval/longmemeval_s_cleaned.json
```

Replay a validated release through one backend and write the same JSONL trace format:

```bash
make eval-external DATASET=longmemeval \
  DATA_PATH=data/longmemeval/longmemeval_s_cleaned.json SYSTEM=scopegraph
```

The external runner uses a credential-free turn-preserving extractor by default. Add
`LIVE_ANSWER=1` to call the configured OpenAI-compatible answer model; retrieval and
answer scoring remain separate.

Use `LIVE=1` for external replay with live extraction, cached real embeddings, and the configured answer model. Run it with `--limit` first because full external releases can require many provider calls.
