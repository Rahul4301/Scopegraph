# Evaluations

CrossScopeMem is the controlled thesis benchmark for account-level scope isolation.
It is reported separately from the official LongMemEval-S, LoCoMo, and
MemoryAgentBench results. External benchmark data is never fabricated and downloaded
artifacts remain ignored by Git.

Run the paired ScopeGraph controls and ablations over account-shaped scenarios:

```bash
make eval-diagnostic
```

Run ScopeGraph's complete live model path (default: 10 scenarios at difficulty 3):

```bash
make eval-diagnostic-live LIVE=1
```

The batch uses one frozen extraction per source and runs full ScopeGraph, vector-only,
flat-graph, two-level session/global, no graph traversal, and no
temporal/status filtering. The evaluation container uses Bolt port `7688` and separate Docker volumes. Its
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

The raw record preserves the scenario, question, gold state, retrieved IDs/scopes/scores,
verbatim provenance messages supplied to the answerer, trace, latency, token count, storage
statistics, configuration hash, seed, and git commit. Retrieval packs memory summaries and
their supporting chat messages into one shared token budget; it never supplies a whole
conversation merely because it was ingested. Scoring is a separate pass. The default answer
field uses a transparent deterministic fact extractor; no LLM answer is claimed.
LoCoMo reports its official token F1 as the primary score and a separate normalized,
order-insensitive exact-match accuracy diagnostic; adversarial questions use the official
binary abstention rule for both.

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

The default run contains 30 injected-error cases and writes both raw JSONL and a
bootstrap-confidence-interval summary. Use `--cases` only for a labeled smoke test or
a pre-registered final sample size.

Fetch and validate every selected official release:

```bash
make download-benchmarks
make validate-benchmarks
```

Run all 6,157 questions through full ScopeGraph with live extraction, embeddings,
answers, and official judges:

```bash
make eval-suite BATCH=results/batches/paper-v1
```

The external runner uses the isolated Neo4j service and a credential-free
turn-preserving extractor by default for local plumbing tests. The paper target uses
`--live`, which enables live extraction, cached real embeddings, live answers, and
the required official judge. Use `--limit 5` only for a labeled question smoke run,
or `--case-limit 1` to run every question from one complete shared history; research
commands run every official question. Sessions after a question timestamp are
excluded, and supplied evidence IDs are retained as retrieval gold. LoCoMo has
no per-question timestamp, so its as-of time is the final observed conversation
turn rather than the machine's current date. Its release session dates are copied
into the stored Session and SourceMessage timestamps; historical sessions close at
their latest stored message time, not import time. The release dates lack timezone
offsets, so the adapter treats them as UTC rather than inventing a local timezone.

`make eval-suite` writes one extraction cache per dataset for reproducibility and safe
resume. A cache is rejected if the dataset hash, extraction model, or LoCoMo adapter
version does not match.

Pass `--resume` with the same output path to continue an interrupted external run;
the checkpoint rejects changes to the dataset bytes, code, configuration, models, or
system.
The runner completes the selected questions and prints the final official score and
failure count. Failed questions receive zero official score and remain in the
denominator; the CLI exits nonzero after reporting if any question failed. A matching
extraction cache can be reused for a fresh run after diagnosing failures.
