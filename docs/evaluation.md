# Evaluation

CrossScopeMem is the controlled, thesis-specific benchmark for scope isolation. Each
scenario is one account containing a global root, simultaneous projects, a nested
repository, standalone conversations, and chronological sessions. Its generated
questions must never be combined with external benchmark scores. LongMemEval-S,
LoCoMo, and MemoryAgentBench supply complementary external-validity evidence using
only their official questions and answers.

Run a fast controlled smoke test (the report target needs a `BATCH` directory):

```bash
make eval-diagnostic
make eval-report BATCH=results/batches/<run-id>
make smoke
```

The proposal target is 40–60 complete accounts. For example:

```bash
make eval-ablation SCENARIOS=40 DIFFICULTY=3
make eval-report BATCH=results/batches/<run-id>
```

`eval-diagnostic` defaults `SCENARIOS` to 1 and `eval-ablation` to the 10-account pilot minimum (it refuses fewer unless `SMALL=1`). Neither makes paid calls unless `LIVE=1` or `LIVE_ANSWER=1` is passed; without them extraction is the oracle, embeddings are a local hash, and no answers are generated. A reportable
run must explicitly select its pre-registered account count.

To run the complete live model pipeline through the same repository:

```bash
make eval-diagnostic-live SCENARIOS=10 DIFFICULTY=3 LIVE=1
```

This starts a second Neo4j Community container on Bolt port `7688`, separate from
the development database on `7687`. The runner clears only that evaluation
database between scenarios because generated scenarios intentionally reuse fixture
IDs. Ten live difficulty-3 scenarios are intended as a roughly 30–40 minute pilot;
provider latency and rate limits can move the wall-clock time outside that range.
The Neo4j run measures ScopeGraph's end-to-end repository path.

The batch freezes extraction once per source and runs seven paired conditions: full
ScopeGraph; vector-only, vector-with-scope-filter, flat-graph, and two-level session/global
controls; and no-graph-traversal and no-temporal/status ablations. Flat graph is also the no-hierarchy
ablation, so it is not executed twice. The raw JSONL
record preserves retrieved IDs, scopes, scores, status, trace paths, latency, token
count, logical storage statistics, configuration hash, seed, and git commit.
`evals.analysis.aggregate` scores raw records independently of execution and resamples
complete accounts for confidence intervals. Reports also write paired full-minus-
control differences and exact McNemar results for binary outcomes.

Implemented aggregate metrics include exact match, normalized token F1, Precision@K, Recall@K, Cross-Scope Contamination Rate, stale-memory rate, p50/p95 latency, token summaries, and logical storage counts. The correction runner measures error relapse after no correction, conversational correction, and direct graph correction at +1, +5, +10, and +20 sessions.
It defaults to 30 injected-error cases, appends real distractor memories between
probes, and reports account/case-bootstrap confidence intervals. A null difference
between conversational and structural correction must be reported if observed.

Live extraction also writes `classification.json` with one gold/predicted pair per
source-memory candidate. It reports candidate coverage, strict and matched-only
scope-level accuracy, strict and matched-only concrete target accuracy, missing and
extra candidates, and a level confusion matrix. Strict accuracy divides by all gold
candidates, so extraction omissions cannot inflate classification quality. A session
target consists of both its containing scope and session ID. Offline oracle extraction
writes the same artifact with `evaluated: false`, because comparing oracle labels with
themselves would be meaningless. `eval-report` copies the artifact into the processed
report and renders `tables/scope_classification.md`.

## Proposed per-question concurrency (not implemented)

A future runner may execute as many as 50 independent question trials at once and
then feed every completed record through the existing scoring pass. This is an
execution optimization, not a change to metrics. A defensible implementation must:

- use an explicit `--question-concurrency` setting with a default of 1 and a hard
  ceiling of 50;
- bound work with an async semaphore, preserve deterministic output order, and
  checkpoint each completed question before scheduling replacements;
- keep chronological ingestion serialized within each scenario, then fan out only
  immutable query snapshots so concurrent questions cannot mutate shared memory;
- use separate limits for memory search, answer generation, and judge calls, because
  each provider has different quotas and latency;
- record queue time separately from retrieval, answer, and grading latency;
- apply per-request timeouts, retry only transient failures with jitter, respect
  `Retry-After`, and never convert exhausted retries into incorrect answers;
- grade only after the corresponding answer is durably checkpointed, with a pinned
  judge/model/prompt across runs; and
- report effective concurrency, rate-limit events, failures, retries, and cost so a
  50-way run cannot be mistaken for a serial latency benchmark.

The current runner does not provide per-question concurrency.

## External benchmarks

No external benchmark results are claimed until the live suite completes. The primary suite
is exactly LongMemEval-S, LoCoMo, and MemoryAgentBench: 6,157 official questions under full
ScopeGraph. Run:

```bash
make download-benchmarks
make validate-benchmarks
make eval-suite BATCH=results/batches/<run-id>
```

Each LoCoMo history (ten shared conversations) and each MemoryAgentBench corpus is ingested
once, not once per question. Live extraction is checkpointed once per official source session
into one artifact per dataset, so an interrupted full run resumes without re-extracting sources.

### LoCoMo runs

`make eval-smoke CASES=N` runs the first N conversations (10 = all 1,986 questions) at the
current commit with a fresh extraction cache and writes a new, never-overwritten
`results/smoke/mm_dd__hh_mm.jsonl` (local time) plus `.run.json`, `.report.md` and
`.questions.csv`. `ABLATION=vector_only` adds
the plain-vector baseline (a documented amendment outside the original protocol; see
[benchmark_protocol.md](benchmark_protocol.md)). Report LoCoMo accuracy both with and without
category 5, which is scored by an abstention rule and says nothing about retrieval:

```bash
make eval-report BATCH=<dir containing the jsonl> LOCOMO_DATA=data/locomo/locomo10.json
```

writes `processed/locomo_diagnostics.json` (per-category accuracy, F1, gold-source recall at
retrieval and at delivery, accuracy when gold was missed, outcome stages) and
`processed/locomo_zero_gold_questions.json` (every question whose annotated evidence turns
were not retrieved, with the turn text).

## Statistics and conditions

- Conditions (seven): full ScopeGraph; `vector_only_control`; `vector_scope_filter` (the vector
  control plus a hard metadata filter on each question's `allowed_scope_ids`); `flat_graph_control`
  (also the no-hierarchy ablation); `two_level_control`; `no_graph_traversal`; `no_temporal_status`.
  `make eval-ablation` runs all seven in one batch.
- Because `vector_scope_filter` filters on the same `allowed_scope_ids` that define
  `cross_scope_contamination`, its contamination is 0 by construction. It tests whether ScopeGraph's
  hierarchy adds anything beyond a correct access filter, not whether a filter prevents leakage.
- Confidence intervals are 95% percentile bootstrap intervals that resample whole accounts
  (CrossScopeMem scenarios, LoCoMo conversations). A pilot needs at least 10 accounts; the proposal
  target is 40–60. Below 10 the report marks results `underpowered`, and with one account it
  emits no interval rather than a degenerate `mean = lower = upper`.
- Paired differences are **full minus control**. For lower-is-better metrics
  (`cross_scope_contamination`, `any_cross_scope_contamination`, `stale_memory_error_rate`) a
  negative difference favours full; otherwise a positive difference favours full.
- `make eval-report` writes `processed/confidence_intervals.json`, `processed/mcnemar.json`, and
  `tables/mcnemar.md`. McNemar rows exist only for per-question 0/1 outcomes (`exact_match`,
  `gold_hit_at_8`, `any_cross_scope_contamination`, binary official judge scores). Questions inside
  one account are correlated, so also read the account-level sign test in the same table.

## Timing

Retrieval, embedding, answer, and judge time are separate fields and are never summed. Reports
write `tables/latency_<dataset>.md` per dataset. `retrieval_*` is the timed retrieval call and
includes embedding lookups made inside it; `retrieval_core_*` subtracts them; `answer_*` and
`judge_*` are provider round trips. CrossScopeMem timings (tens of milliseconds, warm in-process
embeddings) and LoCoMo timings (about a second, live embedding cache lookups over every source
turn) measure different things and must not be compared or averaged.

## Run artifacts

New runs are named by local start time, `mm_dd__hh_mm` (a batch is a directory of that name; a
second run in the same minute gets `_2`). When a run finishes it writes **`report.md`** (the file to
read: headline table, full-vs-control comparison, LoCoMo categories, mistakes, caveats) and
**`questions.csv`** (one row per question for spreadsheets). The raw `.jsonl` is the machine record.
`make eval-report` adds the detailed JSON/tables/figure under `report/`.

Every new run writes `run.json` (batch directories) or `<name>.run.json` (single JSONL files):
commit, dirty flag and diff hash, config hash, models, seed, protocol, selection, and the
reproduce command. New runs also move `retrieved_source_contents` and `delivered_source_contents`
to `<name>.bulk.jsonl`; scoring never reads them. Field definitions: [results/SCHEMA.md](../results/SCHEMA.md).
