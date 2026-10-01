# Implementation audit (2026-09-23)

This is an audit of the checked-in implementation and locally acquired benchmark
files. The research proposal and master prompt are not present in this repository or
its Git history, so this document does not claim a line-by-line proposal comparison.

## Confirmed alignment

- ScopeGraph is the only memory system implemented and exported.
- Production and research-facing evaluation commands use the isolated Neo4j
  evaluation service. The in-memory repository remains only as a unit-test and smoke
  test double.
- Memory records retain scope, temporal state, source-message provenance, graph
  relations, and reversible correction history.
- Raw evaluation records retain configuration and source fingerprints, model names,
  retrieval traces, latency, token counts, and storage statistics.

## Remaining evaluation cautions

1. All three selected complete releases validate and their official scoring protocols
   are wired in, but no full live result should be claimed until `make eval-suite`
   completes and its output is audited.
2. External examples are mapped to one custom scope beneath a global root. That tests
   retrieval over a memory history, but it does not test ScopeGraph's central claim
   about interference among multiple project/context scopes.
3. External corpora contain no meaningful competing-project structure, so architecture
   controls are not run on them. The separate CrossScopeMem account suite supplies the
   required competing scopes and reports vector-only, flat-graph, and two-level controls.
4. Retrieval timing excludes ingestion, extraction, embedding preparation, answer
   generation, and grading. It must not be presented as end-to-end latency.
5. The external runner is serial. Provider retries exist, but question-level
   concurrency and explicit monetary cost accounting do not.

## Engineering limits

- Exact cosine scoring scans every eligible memory in the selected scopes. Very large
  individual scopes need a measured vector-index design before scalability claims.
- The API has no authentication, account/tenant authorization, quotas, distributed workers,
  or production backup policy.

These gaps must be resolved before describing the outputs as official benchmark
scores or as proof of general usefulness. A small, clearly labeled pilot can support
an undergraduate prototype/feasibility claim, but not benchmark parity, superiority,
or broad generalization.

---

# Addendum (2026-10-01): LoCoMo smoke runs of 2026-09-25/26

Appended; the audit above is unchanged. Scope: the six files in `results/smoke/`, all
`external-v7`, live extraction, live embeddings, live answers, isolated Neo4j (`storage=neo4j`),
seed 42. They cover conv-26 (199 questions) or conv-26 + conv-30 (304 questions) of the ten LoCoMo
conversations, so none is a complete LoCoMo result. File times are local (PDT); record timestamps
are UTC, which is why filenames dated 09-25 hold 09-26 timestamps.

| File | N | Recorded commit | Official metric | Headline |
| --- | --- | --- | --- | --- |
| `locomo-case1-live.jsonl` | 199 | `d47d26a` | `locomo_f1` only (no judge) | F1 0.512 |
| `9_25_10-30.jsonl` | 199 | `d47d26a` | `locomo_f1` only (no judge) | F1 0.612 |
| `locomo-case1-judged.jsonl` | 5 | `d47d26a` | `llm_judge_accuracy` | plumbing check only |
| `9_267_11-00.jsonl` | 199 | `d47d26a` | `llm_judge_accuracy` | 0.729 (0.717 without category 5) |
| `9_26_11-30.jsonl` | 199 | `d47d26a` | `llm_judge_accuracy` | 0.734 (0.717 without category 5) |
| `locomo-20260925-235949.jsonl` | 304 | `73b09ca` | `llm_judge_accuracy` | 0.793 (0.790 without category 5) |

Findings:

1. **Commit provenance is weaker than it looks.** `git_commit` is `HEAD` at run time with no dirty
   flag. The rubric judge and LoCoMo correctness fixes were committed afterwards as `73b09ca`, so
   the three `d47d26a` judged runs may have run on an uncommitted tree that cannot be
   reconstructed. `config_hash` includes a source fingerprint, which distinguishes code states but
   cannot be inverted. New runs record the dirty flag and a diff hash in `run.json`.
2. **Judge changed between runs.** The two earliest 199-question runs have no LLM judge. Category
   1–4 answers in later runs are graded by the pinned `gpt-4o-2024-08-06` rubric; category 5 is
   graded by the abstention rule (`judge_model` is null for those rows). The runs are therefore
   not interchangeable, and the earlier ones are marked superseded in `results/README.md`.
3. **Run-to-run noise is at least 0.5 points.** `9_267_11-00` and `9_26_11-30` share a recorded
   commit and differ by 0.005 accuracy; no repeat count supports a tighter bound.
4. **Two conversations cannot support an interval.** Conversation-level bootstrap intervals are
   flagged `underpowered` below 10 conversations.
5. **Gold-evidence retrieval.** In the 304-question run, 52 of 301 questions with annotated
   evidence (17.3%) retrieved none of their gold turns (10.3% when adjacent delivered turns count).
   The 22% figure raised in review does not reproduce from stored records: the three 199-question
   runs give 16.8–17.3%, and 22.5% is the category-5-only rate. Missing the gold turn is a strict
   proxy, not a failure count: answers were still correct about half the time for categories 2–4,
   because other turns carried the fact, or the annotated turn is itself generic (for example
   `conv-26:D1:12`, "You'd be a great counselor! ... take a look at this.").
6. **Category 1 is an aggregation problem.** Category 1 scored 0.512, and only 31% of its questions
   had every annotated gold turn delivered; accuracy was 0.77 when all were delivered and 0.38
   otherwise. Retrieving at least one gold turn (83%) is not enough for questions that need a
   list assembled from several turns.
7. **Latency is not comparable with CrossScopeMem.** Retrieval p50 was about 1.1 s and answer p50
   about 0.9–1.1 s. These runs predate the split between embedding time and the rest of retrieval
   (`retrieval_embedding_ms`), so the share of the 1.1 s spent on embedding lookups over every
   source turn is unmeasured; do not set it beside the ~40 ms CrossScopeMem figure.

---

# Addendum (2026-10-01): evaluation correctness review

1. **Answer model lacked scope and time.** `run_eval.py` called the answerer with only the
   question and evidence. Seven of the 190 answers in the Neo4j-live run
   (`results/raw/20260921T173312958422Z_cross_scope_mem_scopegraph_42.jsonl`) were `UNKNOWN` for
   "What database does this project use normally?" with the gold evidence ranked first. Fixed: the
   answerer receives the current scope and the as-of time (regression test
   `tests/evals/test_answer_context.py`). The recorded exact match (0.963) is affected and is
   superseded until the run is repeated; its retrieval metrics are not affected.
2. **Confidence-interval artifacts.** The existing `confidence_intervals.json` files were written
   by an earlier, since-removed function (`confidence_intervals`, commit `7dd2153`) that computed
   paired bootstrap differences of `scopegraph` minus each control, resampling accounts. The
   negative contamination values are therefore full minus control, i.e. lower contamination for
   full; the convention was not written down. `mean = lower = upper` appears in
   `results/audit-report/` where all three replayed accounts are `smoke`-profile accounts with
   identical per-account differences, so resampling accounts cannot vary; a single-account run would
   behave the same way. The Neo4j-live file is `{}` because that run contains one system and no
   control to compare against. The current report resamples whole accounts, documents the sign
   convention, flags fewer than 10 accounts, and emits no interval for one account.
3. **Documented ablations never appeared in a batch.** The existing 3-account batch
   (`results/batches/20260921T051029657338Z`, protocol `cross-scope-v3`) holds only four systems.
   `run_all.py` already runs `no_graph_traversal` and `no_temporal_status`; a new batch includes both
   plus `vector_scope_filter`, and a test asserts all seven conditions run and that the two
   ablations change results.
4. **A vector baseline with a metadata filter was missing.** Added as `vector_scope_filter`. In the
   offline 40-account batch it matches full ScopeGraph on recall while having zero contamination
   by construction; see `RESULTS.md` before citing any scope-isolation advantage.
5. **Full-system contamination is one question type.** All 3.7% contamination of full ScopeGraph in
   that batch comes from `q_nested_repository`, where parent-scope memories are retrieved by design
   (ancestor fallback) but the question's `allowed_scope_ids` excludes the parent.
