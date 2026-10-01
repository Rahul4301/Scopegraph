# Results

Everything here is output of `evals/`. Treat existing files as immutable: re-running a report
needs a new `--output-root` (or an explicit `--overwrite`), and runs refuse to write to an
existing JSONL. Do not rename or edit anything in this directory; add new runs and reports beside
the old ones. Most of `results/` is git-ignored, so the files cited in
[../RESULTS.md](../RESULTS.md) exist only on the machine that produced them.

Field and metric definitions: [SCHEMA.md](SCHEMA.md). Claims and their caveats:
[../RESULTS.md](../RESULTS.md). `make results-index` re-derives the columns below from the files
and fails if a run on disk is missing from this table.

## Status vocabulary

- **official** — the complete pre-registered protocol (all questions, all accounts) at one clean
  commit with one judge. **No run is official yet.**
- **smoke** — partial, pilot-sized (fewer than the proposal's 40–60 accounts), or offline
  (oracle extraction, hash embeddings). Fine for plumbing and for stating what was observed;
  not a headline result.
- **superseded** — replaced by a later run that fixes a known defect, or produced under a
  protocol version that is no longer in the code.

## Run table

| run_id | date | commit | dataset | systems | seeds | judge | metric | N | status | path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20260920T054404792447Z | 2026-09-20 | 7e05d55 | cross_scope_mem (v2) | scopegraph, vector_memory, flat_graph, two_level_graph | 42, 43, 44 | none | recall_at_8, exact_match | 60 | superseded | `results/batches/20260920T054404792447Z` |
| 20260921T051029657338Z | 2026-09-21 | af2ba6f | cross_scope_mem (v3) | scopegraph, vector_memory, flat_graph, two_level_graph | 42, 43, 44 | none | recall_at_8, exact_match | 228 | smoke | `results/batches/20260921T051029657338Z` |
| replay_20260920T054404792447Z | 2026-09-21 | af2ba6f | cross_scope_mem (v3) | scopegraph, vector_memory, flat_graph, two_level_graph | 42, 43, 44 | none | recall_at_8 | 60 | smoke | `results/audit-report/frozen-replay/replay.jsonl` |
| 20260921T173312958422Z_cross_scope_mem_scopegraph_42 | 2026-09-21 | af2ba6f | cross_scope_mem (v3) | scopegraph (full only) | 42–51 | none | recall_at_8, exact_match | 190 | smoke | `results/raw/20260921T173312958422Z_cross_scope_mem_scopegraph_42.jsonl` |
| 20261001T173040066573Z | 2026-10-01 | 73b09ca (dirty) | cross_scope_mem (v4) | scopegraph × 7 conditions | 42–81 | none | recall_at_8, contamination | 6160 | smoke | `results/batches/20261001T173040066573Z` |
| 20260926T020654Z_locomo_scopegraph | 2026-09-26 | d47d26a | locomo | scopegraph/full | 42 | none | locomo_f1 | 199 | superseded | `results/smoke/locomo-case1-live.jsonl` |
| 20260926T052906Z_locomo_scopegraph | 2026-09-26 | d47d26a | locomo | scopegraph/full | 42 | none | locomo_f1 | 199 | superseded | `results/smoke/9_25_10-30.jsonl` |
| 20260926T061245Z_locomo_scopegraph | 2026-09-26 | d47d26a | locomo | scopegraph/full | 42 | gpt-4o-2024-08-06 | llm_judge_accuracy | 5 | superseded | `results/smoke/locomo-case1-judged.jsonl` |
| 20260926T061814Z_locomo_scopegraph | 2026-09-26 | d47d26a | locomo | scopegraph/full | 42 | gpt-4o-2024-08-06 | llm_judge_accuracy | 199 | superseded | `results/smoke/9_267_11-00.jsonl` |
| 20260926T063843Z_locomo_scopegraph | 2026-09-26 | d47d26a | locomo | scopegraph/full | 42 | gpt-4o-2024-08-06 | llm_judge_accuracy | 199 | superseded | `results/smoke/9_26_11-30.jsonl` |
| 20260926T065950Z_locomo_scopegraph | 2026-09-26 | 73b09ca | locomo | scopegraph/full | 42 | gpt-4o-2024-08-06 | llm_judge_accuracy | 304 | smoke | `results/smoke/locomo-20260925-235949.jsonl` |

The `seeds` column is the scenario seed for CrossScopeMem (one account per seed) and the fixed
label 42 for external runs. `N` counts question records across all conditions in a batch.

## Notes per run

- **20260920… (v2), superseded.** Protocol `cross-scope-v2`, 3 accounts, 15 questions per system;
  the protocol and its system names (`vector_memory`, `flat_graph`, `two_level_graph`) are no
  longer in the code.
- **20260921T05… (v3), smoke.** Live extraction and answers, 3 accounts, four systems only: no
  `vector_scope_filter`, no `no_graph_traversal`, no `no_temporal_status`. Too few accounts for a
  meaningful interval. Its old `report/processed/confidence_intervals.json` came from a removed
  function (full minus control, 3 accounts); read `summary.json` instead and see the 2026-10-01
  audit addendum.
- **replay_…, smoke.** Frozen-extraction replay of the v3 batch with cached vectors and no answers
  (`results/audit-report/`).
- **20260921T1733… (190 questions), smoke.** Live Neo4j run of full ScopeGraph over 10 accounts, no
  controls. Its 0.963 exact match is affected by the answer-model scope/timestamp bug (7
  `UNKNOWN` answers); retrieval metrics are unaffected. Its report's
  `confidence_intervals.json` is `{}` because the run has a single system and nothing to compare.
  Rerun needed (see RESULTS.md).
- **20261001T1730… (new batch), smoke.** Seven conditions × 40 accounts × 22 questions, difficulty 3,
  in-memory repository, oracle extraction, hash embeddings, no answers. The commit hash is the HEAD
  at run time with uncommitted changes (the `run.json` stores the diff hash). Report in
  `report/`.
- **LoCoMo files.** All cover conv-26 (199 questions) or conv-26 + conv-30 (304), never all ten
  conversations. Two have no LLM judge, the others use the pinned rubric judge for categories 1–4
  and the abstention rule for category 5. The 2026-10-01 audit addendum explains why the
  `d47d26a` runs are superseded by the `73b09ca` run. Report for the 304-question run:
  `results/reports/locomo-20260925-235949/`.
- Extraction caches (`*-extractions.json`) beside the smoke runs are frozen inputs, not results.
