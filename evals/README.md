# Evaluations

Phase 7 implements the credential-free CrossScopeMem benchmark and a reproducible evaluation harness. External local-file adapters cover LongMemEval, LongMemEval-V2, LoCoMo, MemConflict, MemoryAgentBench, RHELM, MemBench, Mem2ActBench, and TIME; benchmark data is never committed or fabricated.

Run all four controlled backends:

```bash
make eval-all
```

Run ScopeGraph against an isolated real Neo4j database (default: 10 scenarios at
difficulty 3):

```bash
make eval-neo4j LIVE=1
```

The evaluation container uses Bolt port `7688` and separate Docker volumes. Its
contents are reset between scenarios; the normal application database is not
touched.

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

Each batch also writes `classification.json`. Live extraction compares effective
predicted memory placement against the scenario oracle, including missing and extra
candidates and a session/scope/global confusion matrix. Oracle-extraction runs mark
classification as not evaluated. This ingestion metric is deliberately separate from
retrieval recall and contamination.

For end-to-end answer evaluation with the configured provider, add `LIVE_ANSWER=1`:

```bash
make eval-all LIVE_ANSWER=1
```

Use `--scenario-count` with the module runner to control the number of generated histories. Keep the default deterministic mode for retrieval-only comparisons and use the same live answer model across every architecture when comparing answer accuracy.

For the complete live pipeline, use real extraction, cached OpenAI embeddings, and the configured answer model together:

```bash
make eval-all LIVE=1
```

Run independent backend evaluations concurrently with a bounded worker count:

```bash
make eval-all SCENARIOS=40 DIFFICULTY=3 LIVE=1 LIVE_ANSWER=1 CONCURRENCY=4
```

`CONCURRENCY` may be from `1` to `4`, matching the four backends. Backend runs are
parallelized, while each backend preserves chronological session replay so temporal
snapshots remain valid. Use `CONCURRENCY=1` for serial execution or lower API pressure.

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
answer scoring remain separate. Sessions and turns after a question timestamp are
excluded. When a release supplies evidence session/turn IDs, the runner records them
as retrieval gold. Embeddings are prepared before the timed retrieval region and the
preparation duration is reported separately.

Use `LIVE=1` for external replay with live extraction, cached real embeddings, and the configured answer model. Run it with `--limit` first because full external releases can require many provider calls.

Pass `--resume` with the same output path to continue an interrupted external run;
the checkpoint rejects changes to the dataset bytes, code, configuration, models, or
system.
