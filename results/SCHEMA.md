# Result schema

One JSONL line is one `EvaluationRecord` (`evals/schemas.py`): one question asked of one
system/condition. Scoring is a separate pass (`evals/analysis/aggregate.py::score_record`) that
adds the metrics below; raw lines never contain scores computed after the fact except the
`official_*` fields, which the external runner computes while grading.

Older files may lack newer fields (they load with defaults) and keep bulk text inline; new runs
write `<name>.bulk.jsonl` (see "Sidecar"). `<name>.run.json` / `run.json` carry run provenance.

## Identity and provenance

| Field | Meaning |
| --- | --- |
| `protocol_version` | Protocol/code generation, e.g. `cross-scope-v4/research`, `external-v7`. Files with different values must not be aggregated together. |
| `evaluation_mode` | Which stages were live: `extraction=live\|oracle\|turn-preserving;embeddings=live\|hash;answer=live\|not-evaluated;storage=…`. |
| `latency_protocol` | Timing boundary, e.g. `warm-embeddings/neo4j-repository` (embeddings prepared before the timed call). |
| `run_id` | Run identifier. Batches share a directory name, not necessarily this value. |
| `dataset` | `cross_scope_mem`, `locomo`, `longmemeval`, `memoryagentbench`. |
| `system` | Always `scopegraph` in current code; older files use `vector_memory`, `flat_graph`, `two_level_graph` as systems. |
| `ablation` | Condition: `full`, `vector_only_control`, `vector_scope_filter`, `flat_graph_control`, `two_level_control`, `no_graph_traversal`, `no_temporal_status` (CrossScopeMem); `full`, `vector_only` (LoCoMo amendment). |
| `scenario_id` | The cluster for statistics: a CrossScopeMem account or an external history (LoCoMo conversation id). |
| `question_id`, `question_type`, `question`, `gold_answer` | The question. For LoCoMo category 5, `gold_answer` is the tempting **wrong** answer; the correct response is to abstain. |
| `config_hash` | SHA-256 of config, retrieval settings, models, ablation, storage and a fingerprint of `src/`, `evals/`, `configs/`. Differs between conditions by design. |
| `git_commit` | `HEAD` when the record was written. Says nothing about uncommitted changes; use `run.json` (`git.dirty`, `git.working_diff_sha256`). |
| `seed` | CrossScopeMem scenario seed; the constant 42 on external runs (nothing random is seeded from it). |
| `timestamp` | UTC creation time of the record. |

## Gold (what a correct retrieval would contain)

| Field | Meaning |
| --- | --- |
| `gold_source_ids` | Source messages that contain the evidence (synthetic: by construction; LoCoMo: the dataset's `evidence` turns, namespaced `<conversation>:<turn>`). Empty means no annotation, not "nothing needed". |
| `gold_memory_ids` | CrossScopeMem only: memories whose provenance intersects `gold_source_ids`. Empty on external runs. |
| `gold_scope_ids` | Scopes that hold the evidence. |
| `allowed_scope_ids` | Scopes the asker may see (current scope, ancestors, global, and any scope the question names). Defines contamination and, in `vector_scope_filter`, the search filter. |
| `current_scope_id` | The scope the question is asked from. |

## Retrieval output (parallel lists, one entry per returned item)

| Field | Meaning |
| --- | --- |
| `retrieved_memory_ids`, `retrieved_contents`, `retrieved_scope_ids`, `retrieved_statuses`, `retrieval_scores` | Item id (`source:<id>` for raw source turns), text, containing scope, status (`active`, `superseded`, …) and final score. |
| `retrieved_source_ids` | Per item, the source message ids linked as provenance (the basis of source-based recall). |
| `retrieved_origin_scope_ids` | Per item, the scopes the item's source messages were written in. Contamination uses this, not the scope an item is stored under, so a global memory built from another project is visible. |
| `delivered_source_ids` | External runs only: per item, ids of the verbatim messages actually shown to the answer model (a retrieved turn plus neighbours, trimmed to budget). Empty for CrossScopeMem. |
| `retrieved_source_contents`, `delivered_source_contents` | Text of the above; bulk, moved to the sidecar in new runs. Not used by scoring. |
| `retrieved_tokens` | Estimated token count of the packed context (`scopegraph.observability.token_counting`, a regex tokenizer, not the model's tokenizer). |
| `trace` | Ranking components and traversal path for each selected item. |
| `storage_stats` | Logical counts: `nodes`, `edges`, `memory_count`, `embedding_count`, `logical_bytes` (serialized memories, CrossScopeMem only; 0 on external runs). Not on-disk size. |

## Timing (milliseconds; never summed)

| Field | Meaning | Does not include |
| --- | --- | --- |
| `retrieval_latency_ms` | Wall time of the retrieval call, including any embedding lookups inside it. | Ingestion, extraction, pre-warming, answering, judging. |
| `retrieval_embedding_ms` | Time inside the embedder during that call. `retrieval_latency_ms − retrieval_embedding_ms` is the retrieval core. `null` before this field existed. | Pre-warm embedding. |
| `embedding_preparation_ms` | Pre-warming of question and memory vectors, outside the timed call. | |
| `answer_latency_ms` | Answer-model round trip (CrossScopeMem: elapsed minus retrieval, so it also contains packing glue). | Retrieval, judging. |
| `judge_latency_ms` | Judge round trip (external, LLM-judged questions only). | |

CrossScopeMem timings (in-process, warm embeddings) and external timings (live embedding cache,
network answer model) are different measurements; compare neither across datasets nor as a
total latency.

## Answer and official score

| Field | Meaning |
| --- | --- |
| `answer`, `hypothesis` | Model output (same text; `hypothesis` is the name the external graders use). `null` if answers were not evaluated. |
| `answer_evaluated` | Whether a live answer model ran. If false, no answer metric is valid. |
| `input_tokens`, `output_tokens` | Provider-reported usage of the answer call. |
| `official_metric`, `official_score` | The benchmark's own grade for this question, e.g. `llm_judge_accuracy` (0/1), `locomo_f1`, `substring_exact_match`. |
| `official_secondary_scores` | Extra official scores, e.g. `locomo_f1` beside the judge. |
| `judge_model`, `judge_input_tokens`, `judge_output_tokens` | Pinned judge and its usage. `null` when the question was graded by rule (LoCoMo category 5). |
| `token_usage` | Per-question extraction, embedding and answer token deltas. |
| `benchmark_metadata` | Dataset fields, e.g. LoCoMo `category` and `evidence_turn_ids`. |
| `failure_type`, `failure_message` | Set when the question raised; it is scored 0 and stays in the denominator. |

## Metrics added by scoring

`k` is 8. "Per question" values are averaged over questions, and intervals resample accounts.

| Metric | Formula | Does not measure |
| --- | --- | --- |
| `recall_at_8` | gold sources found in the union of `retrieved_source_ids[:8]` ÷ `len(gold_source_ids)`; falls back to memory ids if there is no source gold; 0 when there is no gold; `null` for abstention. | Whether the answer was right; whether non-gold evidence also answers the question. |
| `precision_at_8` | items among the first 8 whose sources intersect gold ÷ items returned (≤ 8). Low values are expected: most returned items are context. | Answer quality; it falls when the system returns useful extra context. |
| `gold_hit_at_8` | 1 if any gold source is in the first 8 items, else 0 (`null` for abstention or no gold). A per-question binary used for McNemar. | How many of several needed gold turns were found. |
| `cross_scope_contamination` | share of returned items with **any** origin scope outside `allowed_scope_ids` (falls back to the stored scope if origins are absent). | Whether the answer used the leaked item. Zero by construction for `vector_scope_filter`, and meaningless on external runs (one scope). |
| `any_cross_scope_contamination` | 1 if any returned item in the first 8 has an origin outside `allowed_scope_ids`, else 0. Binary form for McNemar. | As above. |
| `stale_memory_error_rate` | share of returned items whose status is not `active`; `null` for historical questions. | Whether stale text reached the answer. |
| `exact_match` | normalized (lowercase, alphanumerics) answer equals normalized gold. Only with `answer_evaluated`. | Meaning: paraphrases score 0. **Not valid for LoCoMo** (category 5 gold is the wrong answer). |
| `token_f1` | token-overlap F1 of normalized answer and gold. | Same caveats as `exact_match`. |
| `official_<metric>` | The record's `official_score` / secondary scores (failed questions count as 0). | |

LoCoMo specifics: `llm_judge_accuracy` is binary for categories 1–4 (pinned `gpt-4o-2024-08-06`
rubric, `LOCOMO_JUDGE_PROMPT`) and, for category 5, 1 if the answer abstains
("no information available" / "not mentioned"). `locomo_f1` follows LoCoMo's category-aware F1.
Report accuracy with and without category 5. Gold-evidence recall is computed per category in
`processed/locomo_diagnostics.json`.

## Statistics (processed/)

| File | Content |
| --- | --- |
| `summary.json` | Per system/condition metric means; `<metric>_ci95_low/high` (omitted for one account); `account_count`; `query_count`; latency percentiles per stage. |
| `confidence_intervals.json` | Intervals per system and **full − control** differences with `favours`, `underpowered` (< 10 accounts) and an account-level sign test. |
| `paired_ablation_comparisons.json` | Flat form of the paired differences and McNemar counts. |
| `mcnemar.json`, `tables/mcnemar.md` | Exact McNemar for binary outcomes; counts of questions where only full or only the control succeeded. |
| `latency.json`, `tables/latency_<dataset>.md` | Per-dataset latency by stage. |
| `locomo_diagnostics.json`, `locomo_zero_gold_questions.json` | LoCoMo per-category accuracy, F1, gold recall, outcome stages, and unretrieved-evidence questions. |

Bootstrap: 10,000 percentile resamples of whole accounts, seed 42, 95%. Differences are full
minus control; for lower-is-better metrics (contamination, stale rate) negative favours full.

## Sidecar and run.json

- `<name>.bulk.jsonl`: one line per question with `scenario_id`, `question_id`,
  `retrieved_source_contents`, `delivered_source_contents`. Join on `(scenario_id, question_id)`.
  Created only for new runs; a legacy file resumed with `--resume` keeps its text inline.
- `run.json` (batch directory) or `<name>.run.json` (single JSONL): `run_id`, `kind`, `status`
  (`running` means unfinished or crashed), dataset and its SHA-256, `git.{commit,dirty,
  working_diff_sha256}`, `config_hash`, `models`, `seed`, `protocol`, `evaluation_mode`,
  `selection`, `reproduce` (the command line), `python`, and completion counts.
