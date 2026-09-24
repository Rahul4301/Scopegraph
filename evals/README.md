# Evaluations

CrossScopeMem is a credential-free synthetic diagnostic used by tests and retrieval
development. It is not a primary research benchmark. Reported evaluations use the
official questions from LongMemEval-S, LoCoMo, and MemoryAgentBench. Benchmark data
is never fabricated and downloaded artifacts remain ignored by Git.

Run the ScopeGraph evaluation batch:

```bash
make eval-diagnostic
```

Run ScopeGraph's complete live model path (default: 10 scenarios at difficulty 3):

```bash
make eval-diagnostic-live LIVE=1
```

The evaluation container uses Bolt port `7688` and separate Docker volumes. Its
contents are reset between scenarios; the normal application database is not
touched.

Run the synthetic ScopeGraph diagnostic directly when debugging retrieval:

```bash
NEO4J_URI=bolt://localhost:7688 NEO4J_PASSWORD=scopegraph-eval \
PYTHONPATH=src uv run python -m evals.runners.run_eval \
  --dataset cross_scope_mem --config configs/experiments.yaml --allow-neo4j-reset
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
make eval-diagnostic-live LIVE_ANSWER=1
```

Use `--scenario-count` with the module runner to control the number of generated histories. Keep the default deterministic mode for retrieval-only regression checks and pin the live answer model for reported runs.

For the complete live pipeline, use real extraction, cached OpenAI embeddings, and the configured answer model together:

```bash
make eval-diagnostic-live LIVE=1
```

The correction-persistence experiment compares no correction, conversational correction, and direct graph correction at +1, +5, +10, and +20 sessions:

```bash
PYTHONPATH=src uv run python -m evals.runners.run_correction_eval
```

Fetch and validate every selected official release:

```bash
make download-benchmarks
make validate-benchmarks
```

Run all 6,157 questions under all three ablations with live extraction, embeddings,
answers, and official judges:

```bash
make eval-suite BATCH=results/batches/paper-v1
```

The external runner uses the isolated Neo4j service and a credential-free
turn-preserving extractor by default for local plumbing tests. The paper target uses
`--live`, which enables live extraction, cached real embeddings, live answers, and
the required official judge. There is no CLI subset flag: research commands run every
official question. Sessions after a question timestamp are excluded, and supplied
evidence IDs are retained as retrieval gold.

Pass `--resume` with the same output path to continue an interrupted external run;
the checkpoint rejects changes to the dataset bytes, code, configuration, models, or
system.
