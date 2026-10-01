# Results ledger

This file lists every number ScopeGraph currently reports, where it comes from, and what it does
not show. It makes no claim that ScopeGraph is better than, or ready to replace, any other system.
`make claims-check` verifies the rows, the files they cite, and the machine-checked values below.
Run status and file locations: [results/README.md](results/README.md). Field definitions:
[results/SCHEMA.md](results/SCHEMA.md). Most `results/` files are git-ignored, so the cited files
exist only where they were produced.

Two kinds of evidence are kept apart and must not be combined:

- **Scope-isolation claims** come from **CrossScopeMem only**: a generated, synthetic benchmark
  built for this thesis. It can test whether explicit scopes reduce cross-account retrieval errors
  and whether components matter; it cannot show the system works on real conversations.
- **Retrieval-quality claims** come from **external benchmarks (LoCoMo so far)** with their own
  questions, answers and graders. LoCoMo has no competing projects, so it says nothing about scope
  isolation.

Status: `supported` = the number is reproducible from the cited file under the stated caveat;
`partial` = true as stated but narrower than it sounds; `unsupported` = no evidence on disk;
`superseded` = known defect, awaiting rerun.

## Ledger

| id | claim | number | source | command | caveat | status |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | CrossScopeMem, offline: full ScopeGraph returns less out-of-scope evidence than the vector-only control | contamination 0.037 vs 0.769 (40 accounts, 880 questions) | `results/batches/20261001T173040066573Z/report/processed/summary.json` | `make eval-ablation STORAGE=memory SCENARIOS=40 DIFFICULTY=3` then `make eval-report BATCH=results/batches/20261001T173040066573Z` | Synthetic accounts; oracle extraction, hash embeddings, in-memory repository, no answer model; one commit with uncommitted edits (see run.json); seeds 42-81 | partial |
| A2 | Paired difference, full minus vector_only_control: lower contamination and higher Recall@8 | contamination -0.732, Recall@8 +0.163; sign agrees in all 40 accounts | `results/batches/20261001T173040066573Z/report/processed/confidence_intervals.json` | same as A1 | Intervals resample accounts but the accounts are template-generated, so they understate real-world variance; offline only | partial |
| A3 | Against a vector search with a correct metadata filter (vector_scope_filter), full ScopeGraph shows no retrieval advantage | Recall@8 difference +0.002, interval includes 0; filter contamination 0 | `results/batches/20261001T173040066573Z/report/processed/confidence_intervals.json` | same as A1 | Filter contamination is 0 by construction because the filter uses the same allowed_scope_ids as the metric; the filter is given oracle allowed scopes. Tests hierarchy vs filter, not leakage | supported |
| A4 | Ablations, offline: removing graph traversal changes Recall@8 by -0.001 (interval includes 0); removing temporal/status handling raises the stale-memory rate by 0.018 | full minus no_temporal_status stale rate -0.018 | `results/batches/20261001T173040066573Z/report/processed/confidence_intervals.json` | same as A1 | No evidence that graph traversal helps on these questions; no_temporal_status also slightly raises Recall@8 (+0.002 for the control) | partial |
| A5 | Live Neo4j run of full ScopeGraph, 10 accounts | Recall@8 1.000, contamination 0.000, exact match 0.963 | `results/neo4j-live-report/processed/summary.json` | `make eval-diagnostic-live SCENARIOS=10 DIFFICULTY=3 LIVE=1` (original command not recorded) | No control in this run, so it shows nothing about isolation; exact match is affected by the answer-context bug (7 UNKNOWN answers) and is superseded until rerun | superseded |
| A6 | Live extraction placed candidates at the right level and target | scope-level accuracy 1.000 over 1000 gold candidates (26 extra candidates) | `results/neo4j-live-report/processed/scope_classification.json` | same as A5 | Gold placement comes from the generator's templates, so this measures agreement with synthetic labels | partial |
| A7 | ScopeGraph reduces contamination versus vector retrieval with live extraction and embeddings at 10 or more accounts | none | none | `make eval-ablation SCENARIOS=10 LIVE=1` (not yet run) | The only live batch has 3 accounts and four systems | unsupported |
| A8 | Any result at the proposal's 40-60 accounts with live extraction and answers | none | none | `make eval-ablation SCENARIOS=40 LIVE=1 LIVE_ANSWER=1` (not yet run) | Offline 40-account batch exists (A1-A4) | unsupported |
| B1 | LoCoMo answer accuracy, conv-26 and conv-30 only | 0.793 all categories, 0.790 without category 5 (304 questions) | `results/reports/locomo-20260925-235949/processed/locomo_diagnostics.json` | `make eval-smoke CASES=2` (original command not recorded); report: `uv run python -m evals.analysis.run_report results/smoke/locomo-20260925-235949.jsonl --output-root results/reports/locomo-20260925-235949 --locomo-data data/locomo/locomo10.json` | 2 of 10 conversations; pinned gpt-4o rubric judge for categories 1-4, abstention rule for 5; one run, so run-to-run noise is at least 0.005 (see audit); interval flagged underpowered; commit recorded without a dirty flag | partial |
| B2 | LoCoMo category 1 is limited by evidence aggregation, not answer wording | category 1 accuracy 0.512; all annotated gold turns delivered for 31% of category-1 questions | `results/reports/locomo-20260925-235949/processed/locomo_diagnostics.json` | same as B1 | 43 questions; gold-turn annotation is a strict proxy; accuracy 0.77 when all gold delivered vs 0.38 otherwise | partial |
| B3 | Share of LoCoMo questions whose annotated evidence turns were not retrieved | 17.3% overall (52 of 301 annotated); 22.5% within category 5 only | `results/reports/locomo-20260925-235949/processed/locomo_diagnostics.json` | same as B1 | The 22% figure sometimes quoted is the category-5-only rate; many such questions are still answered correctly | partial |
| B4 | LoCoMo result over all 10 conversations (1,986 questions) | none | none | `make eval-smoke CASES=10` (not yet run) | Requires approval; see the plan in the Phase 1 summary | unsupported |
| B5 | ScopeGraph compared with a plain vector baseline on LoCoMo | none | none | `make eval-smoke CASES=10 ABLATION=vector_only` (not yet run) | Baseline is a protocol amendment outside the original protocol | unsupported |
| B6 | LongMemEval-S or MemoryAgentBench results | none | none | `make eval-suite BATCH=results/batches/<run-id>` (not yet run) | No external result exists for either benchmark | unsupported |
| C1 | CrossScopeMem retrieval time, Neo4j, embeddings warmed outside the timer | p50 40.4 ms (190 questions, full ScopeGraph) | `results/neo4j-live-report/processed/summary.json` | same as A5 | Excludes extraction, embedding, answering; in-process embeddings | supported |
| C2 | LoCoMo retrieval and answer time, reported separately | retrieval p50 1114 ms, answer p50 1092 ms (304 questions) | `results/reports/locomo-20260925-235949/processed/summary.json` | same as B1 | Not comparable with C1; retrieval includes embedding-cache lookups over every source turn and these runs predate the embedding-time field, so that share is unmeasured | partial |
| D1 | Direct graph correction persists better than conversational correction | none | none | `PYTHONPATH=src uv run python -m evals.runners.run_correction_eval --allow-neo4j-reset` (not yet run) | The runner exists; no result file exists | unsupported |
| E1 | Production readiness, security, or scalability | none | none | none | Not claimed: the API has no authentication or tenancy, and exact-cosine scoring has no scalability measurement (docs/audit.md) | unsupported |

## Machine-checked values

`make claims-check` reads each file and compares the value (absolute tolerance 1e-6, relative for
large numbers).

| id | file | path | expected |
| --- | --- | --- | --- |
| A1 | `results/batches/20261001T173040066573Z/report/processed/summary.json` | `scopegraph.cross_scope_contamination` | 0.036800231 |
| A1 | `results/batches/20261001T173040066573Z/report/processed/summary.json` | `scopegraph/vector_only_control.cross_scope_contamination` | 0.769034091 |
| A2 | `results/batches/20261001T173040066573Z/report/processed/confidence_intervals.json` | `paired_differences_full_minus_control.cross_scope_mem/scopegraph/vector_only_control.metrics.cross_scope_contamination.mean_difference` | -0.73223386 |
| A2 | `results/batches/20261001T173040066573Z/report/processed/confidence_intervals.json` | `paired_differences_full_minus_control.cross_scope_mem/scopegraph/vector_only_control.metrics.recall_at_8.mean_difference` | 0.163293651 |
| A3 | `results/batches/20261001T173040066573Z/report/processed/confidence_intervals.json` | `paired_differences_full_minus_control.cross_scope_mem/scopegraph/vector_scope_filter.metrics.recall_at_8.mean_difference` | 0.00158730159 |
| A3 | `results/batches/20261001T173040066573Z/report/processed/summary.json` | `scopegraph/vector_scope_filter.cross_scope_contamination` | 0 |
| A4 | `results/batches/20261001T173040066573Z/report/processed/confidence_intervals.json` | `paired_differences_full_minus_control.cross_scope_mem/scopegraph/no_temporal_status.metrics.stale_memory_error_rate.mean_difference` | -0.0178474452 |
| A4 | `results/batches/20261001T173040066573Z/report/processed/confidence_intervals.json` | `paired_differences_full_minus_control.cross_scope_mem/scopegraph/no_graph_traversal.metrics.recall_at_8.mean_difference` | -0.000793650794 |
| A5 | `results/neo4j-live-report/processed/summary.json` | `scopegraph.exact_match` | 0.963157895 |
| A5 | `results/neo4j-live-report/processed/summary.json` | `scopegraph.recall_at_8` | 1 |
| A6 | `results/neo4j-live-report/processed/scope_classification.json` | `summary.scope_level_accuracy` | 1 |
| B1 | `results/reports/locomo-20260925-235949/processed/locomo_diagnostics.json` | `scopegraph/full.all_categories.judge_accuracy.mean` | 0.792763158 |
| B1 | `results/reports/locomo-20260925-235949/processed/locomo_diagnostics.json` | `scopegraph/full.excluding_category_5.judge_accuracy.mean` | 0.789699571 |
| B2 | `results/reports/locomo-20260925-235949/processed/locomo_diagnostics.json` | `scopegraph/full.categories.1.judge_accuracy` | 0.511627907 |
| B2 | `results/reports/locomo-20260925-235949/processed/locomo_diagnostics.json` | `scopegraph/full.categories.1.all_gold_delivered_rate` | 0.30952381 |
| B3 | `results/reports/locomo-20260925-235949/processed/locomo_diagnostics.json` | `scopegraph/full.categories.5.zero_gold_retrieved_rate` | 0.225352113 |
| C1 | `results/neo4j-live-report/processed/summary.json` | `scopegraph.retrieval_p50_ms` | 40.407334 |
| C2 | `results/reports/locomo-20260925-235949/processed/summary.json` | `scopegraph.retrieval_p50_ms` | 1114.07967 |
| C2 | `results/reports/locomo-20260925-235949/processed/summary.json` | `scopegraph.answer_p50_ms` | 1091.77062 |
| B3 | `results/reports/locomo-20260925-235949/processed/locomo_diagnostics.json` | `scopegraph/full.gold_evidence_all_categories.zero_gold_retrieved_rate` | 0.172757475 |

## Unsupported or open

A7, A8, B4, B5, B6, D1 and E1 above have no evidence. A5's exact match is superseded. The requested
Phase 1 reruns (A7/A8 live batch, B4/B5 full LoCoMo with the vector baseline, a repeat of A5 with the
answer-context fix) need explicit approval before any paid call.
