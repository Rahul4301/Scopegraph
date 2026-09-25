# Evaluation

CrossScopeMem is the controlled, thesis-specific benchmark for scope isolation. Each
scenario is one account containing a global root, simultaneous projects, a nested
repository, standalone conversations, and chronological sessions. Its generated
questions must never be combined with external benchmark scores. LongMemEval-S,
LoCoMo, and MemoryAgentBench supply complementary external-validity evidence using
only their official questions and answers.

Run a fast controlled smoke test:

```bash
make eval-diagnostic
make eval-report
make smoke
```

The proposal target is 40–60 complete accounts. For example:

```bash
make eval-diagnostic SCENARIOS=40 DIFFICULTY=3
make eval-report
```

`SCENARIOS` defaults to 1 to prevent an accidental expensive live run. A reportable
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

The batch freezes extraction once per source and runs six paired conditions: full
ScopeGraph; vector-only, flat-graph, and two-level session/global controls; and
no-graph-traversal and no-temporal/status ablations. Flat graph is also the no-hierarchy
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

No external benchmark results are claimed until the live suite completes. The primary
suite is exactly LongMemEval-S, LoCoMo, and MemoryAgentBench: 6,157 official questions
under full ScopeGraph. Run `make
download-benchmarks`, `make validate-benchmarks`, then `make eval-suite
BATCH=results/batches/<run-id>`. LoCoMo's ten shared histories and each
Each LoCoMo history and MemoryAgentBench corpus is ingested once, not once per
question. Live extraction is checkpointed once per official source session into one
artifact per dataset so an interrupted full run resumes without re-extracting sources.
