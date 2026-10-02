# Pre-registration v1

Study: does a model need an external memory layer when the whole history fits in context?
Branch: `memstudy`. Written 2026-10-01, before any paid call. Config hash at writing:
`51159301c46ea0e0` (`configs/study.yaml` plus the prompt texts in `src/memstudy/prompts.py`).
Every value below is also in `configs/`; a change after tag `prereg-v1` needs a new tag.

## 1. Hypothesis and falsification

H: full-history prompting with prompt caching matches or beats a memory layer whenever the
history fits in the window, and a memory layer wins only past a token-cost or latency threshold.

H is falsified if either holds:

- F1. A memory layer (arm B) beats full context (arm A) at history lengths that fit in the
  window: the 95% CI of (B minus A) accuracy excludes zero in favour of B on a primary benchmark.
- F2. No cost crossover appears: for no measured history length and queries-per-history Q does a
  memory arm cost less than arm A in total (ingestion plus queries).

Non-inferiority (supports H): arm A is non-inferior to arm B if the lower bound of the 95% CI of
(A minus B) accuracy is above minus 3 percentage points. The margin is 3 points (proposed in the
brief, adopted here).

## 2. Fixed setup

| Role | Model | Settings |
| --- | --- | --- |
| Reader, all arms | `gpt-6-luna` | reasoning effort `none`, temperature 0, max output 512, same template |
| Judge (LoCoMo, LongMemEval-S) | `gpt-5-nano` | reasoning effort `minimal`, max output 256, one fixed prompt |
| Coding grader | the benchmark's own tests | no model |

Phase 0 confirmed (all 2026-10-01, sources in `configs/prices.yaml`):

- GPT-6 Luna exists as `gpt-6-luna`. Context window 1,050,000, max output 128,000. Per 1M tokens:
  input $0.10, cached input $0.01, cache write $0.125, output $0.50. Prompts above 272,000 input
  tokens are priced at 2x input and cache rates and 1.5x output for the whole request. No
  fallback to GPT-5.6 Luna is needed. Account access is not yet confirmed; the first pilot call
  confirms it.
- Pinning: the model page lists only the alias, no dated snapshot. The snapshot actually served is
  recorded from `response.model` on every call and reported. This is a limitation of the pin.
- Temperature 0: verified live on 2026-10-01 (`memstudy api-check`, two calls, $0.0003). With
  reasoning effort `none` the API accepted `temperature=0`; two identical calls returned the same
  answer; `response.model` was `gpt-6-luna`. The explicit cache breakpoint worked as designed:
  the first call reported 2,125 `cache_write_tokens`, the second 2,125 `cached_tokens`, and the
  billed cost matched the price formula to the last digit. Reported cached counts were not
  rounded to a multiple of 128 in this case. With reasoning active, temperature is reportedly
  rejected; the study never enables reasoning for the reader. If the API ever rejects the
  parameter the code raises; it never drops it silently.
- GPT-5 nano: $0.05 input, $0.005 cached, $0.40 output per 1M. Context 400,000. The dated
  snapshot `gpt-5-nano-2025-08-07` is marked Deprecated on its model page, so the alias is used
  and the served snapshot is logged. "Lowest reasoning effort" is `minimal`; the first pilot call
  checks that `minimal` is accepted.
- Caching rules (GPT-5.6 and later, including GPT-6): minimum cacheable prefix 1,024 tokens; cache
  write 1.25x input; cached read 0.1x input; lifetime at least 30 minutes after the last write or
  reuse, refreshed on reuse; `cached_tokens` is rounded down to a multiple of 128. Usage reports
  `input_tokens_details.cached_tokens` and `cache_write_tokens`. Explicit mode with one
  breakpoint writes only the marked prefix; with no breakpoint nothing is written.

Model rule and gate G1. Owner decision: only GPT-6 Luna (reader and Mem0 fact extraction) and
GPT-5 nano (judge) are used as chat models. Mem0's own default extraction models (`gpt-5-mini`
in the library, `gpt-4o-mini` in the benchmark harness) are therefore replaced by `gpt-6-luna`
with reasoning effort `none`; this is a deliberate departure from Mem0's defaults. Neither model
can produce embeddings, so arms B and C still need one embedding model, `text-embedding-3-small`
(Mem0's default, reused for arm C so retrieval is the only difference). The code refuses to run
arms B or C until the owner sets `g1_extra_models: true` in `configs/approvals.yaml`, which
approves that embedding model.

## 3. Arms

- A, no memory: full rendered history first, question last. Explicit cache breakpoint after the
  history when the run queries that history at least twice; no breakpoint for a single query,
  because a write with no reuse is a pure 25% surcharge (LongMemEval-S has one question per
  history). Both modes are priced in the Phase 4 crossover analysis.
- B, Mem0: `mem0ai==2.2.1` open source (April 2026 algorithm: single-pass ADD-only extraction,
  entity linking, multi-signal retrieval). `top_k=200`, `threshold=0.1`, `rerank=false`
  (library defaults except `top_k`). `top_k=200` is the harness default and Mem0's headline
  setting, chosen so the arm is not handicapped. Local Qdrant store. Ingestion: 10 turns per
  `add` call, session date written into each message (the OSS `timestamp` argument is
  Platform-only). `infer=True`. Telemetry off. spaCy `en_core_web_sm` must be installed; without
  it Mem0 silently degrades to semantic-only retrieval, so the arm refuses to start.
  Mem0 Platform (hosted) is not used: its models cannot be pinned and its proprietary
  optimizations are not in the OSS library, so OSS numbers are not expected to match Mem0's
  published ones.
- C, plain RAG (control): verbatim chunks of whole turns, 256 tokens, same embedder as B,
  cosine top 28 chunks (about 7K tokens, matching Mem0's reported mean retrieved tokens), ranked
  by similarity. Oversized turns are split, never cut.
- D, Supermemory: excluded. Its extraction and embedding models run server side and cannot be
  pinned or priced, its pricing was not confirmed in Phase 0, and the brief requires a clean
  setup. Gate: reconsider only if the owner supplies a confirmed price and a pin.

Mem0's self-reported scores are never used as a baseline. Every number comes from a run with a
`run.json`.

## 4. Benchmarks

- LoCoMo (`data/locomo/locomo10.json`): 10 conversations. Primary: categories 1 to 4,
  1,540 questions (verified in the loader test). Reported separately: 446 category 5
  adversarial questions, graded for abstention.
- LongMemEval-S (`longmemeval_s_cleaned.json`): 500 questions, one haystack each, 30 abstention.
- Coding benchmark rule result: VibeMemBench was not usable. Its code repository
  (`AlibabaResearch/DAMO-ConvAI/tree/main/VibeMemBench`) contains only a README reading
  "Coming", and no trajectories are published (checked 2026-10-01). Fallback: SWE Context Bench
  (arXiv 2602.08316, Hugging Face `jiayuanz3/SWEContextBench`, MIT). Caveats: it also ships no
  trajectories, so the memory history is each prior task's issue plus gold patch (a verified
  experience, not an agent trajectory). Grading uses the benchmark's own evaluation system, not
  the stock `swebench` package: the benchmark repository ships a fork of the SWE-bench harness
  (`swebench_memory`, run through `combine_instances` then `run_evaluation`, as its
  `evaluation.sh` does) with its own Docker images (`jiayuanz3/swecontextbench`). Stock
  `swebench` 5.0.2 cannot grade these rows (it needs `image`, `eval_script`, `log_parser`
  fields). A harness failure is recorded as an error, never as unresolved. Cloning the benchmark
  repository at a pinned commit and pulling the images need the owner's approval. The related-task count
  differs between the paper (376) and the dataset README (362); the loaded file is authoritative.
  Arms: memory off versus Mem0 memory on; RAG only if the pilot shows budget room.
- Full sets, no slices, one seed (0).

Token census (Phase 0, `results/phase0/token_census.json`, o200k_base proxy tokenizer):

| Benchmark | Histories | Tokens min / mean / max | Exceed window | Over 272K threshold |
| --- | --- | --- | --- | --- |
| LoCoMo | 10 | 10,965 / 17,968 / 21,173 | 0 | 0 |
| LongMemEval-S | 500 | 97,338 / 103,791 / 106,056 | 0 | 0 |

Every history fits with room to spare and none crosses the long-context price threshold. The
tokenizer for `gpt-6-luna` is not mapped in tiktoken 0.14.0, so counts are a proxy with a 5%
safety margin; the API's own `usage.input_tokens` is authoritative for cost. Nothing is truncated:
an item that does not fit is recorded as `does_not_fit` and reported.

## 5. Mem0 scoping (fairness)

- `user_id = "<bench>_<history_id>"`: one per LoCoMo conversation, one per LongMemEval-S haystack.
  Never shared across test cases. `run_id` is unused (no benchmark defines per-session runs).
- Every Mem0 `add` and `search` carries only that `user_id` in `filters`.
- Test `tests/memstudy/test_mem0_isolation.py` runs the real Mem0 class on a real local Qdrant
  store with an embedder that gives every text the same vector (worst case for leakage) and shows
  that a search under one `user_id` returns zero memories from another.
- Coding scope mapping: one history per repository, `user_id = "swectx_<owner>__<repo>"`. A
  related task queries only its own repository's history. The history is the repository's
  experience-task pool (issue plus gold patch), never the related task itself (the loader raises
  if a task is in both sets). Memories from repo A are never retrievable in repo B.
  `agent_id` is unused.

## 6. Judge

One fixed prompt for every arm, text in `src/memstudy/prompts.py` (hashed into the config hash).
Instruction: grade the model answer against the gold answer and reply with JSON
`{"verdict": "CORRECT"}` or `{"verdict": "INCORRECT"}`. Category guidance, chosen by question
category only (never by arm): default (key information present, contradiction is incorrect);
temporal (any date format, off-by-one on day, week, or month counts is correct); knowledge
update (must give the most recent value as current); preference (uses the stated preferences
described by the rubric); abstention (correct only if it says the information is not mentioned).
Unparseable output is recorded and counted as ungraded, never as correct. Reasoning tokens are
logged. The judge differs from each benchmark's default judge, which is a stated limitation.

Hand-check (Phase 2, pre-registered pass criteria): 100 graded answers, 50 each from arms A and B,
stratified across benchmark and category, shown to the owner without the judge verdict or the arm.
Pass requires: judge agreement with the owner at least 95% overall, and the two arms'
disagreement rates within 2 percentage points. Also reported: the judge's flip rate when rerun
on the same 100 items. If the check fails, stop and report; the judge is not changed without
approval.

## 7. Statistics plan

- Accuracy per arm, benchmark, and category. LoCoMo primary and adversarial reported separately.
- Paired comparison per question: McNemar exact test, A versus B and A versus C, per benchmark.
  Four primary tests, Holm-adjusted. Category breakdowns are exploratory.
- Confidence intervals: 10,000 bootstrap resamples (seed 0) that resample whole histories. For
  LongMemEval-S a history is one question. For LoCoMo only 10 histories exist, so its interval
  is wide; this is stated wherever LoCoMo is reported.
- Non-inferiority: lower 95% bound of (A minus B) against minus 3 points, per benchmark.
- Differences are reported as A minus B. Tokens, dollars, and latency per query from `run.json`.
- Latency compares query-time latency only (retrieval plus reader). Ingestion time is reported
  separately.

## 8. Cost model and crossover

`src/memstudy/costmodel.py`, with queries per history Q as an explicit variable:

- A, cached: `H*write + (Q-1)*H*read + Q*(q*in + a*out)`. A, uncached: `Q*(H*in + q*in + a*out)`.
- B and C: `ingest(H) + Q*((ctx+q)*in + a*out + retrieval)`.
- Crossover Q where the memory arm becomes cheaper in total, per history length, with and
  without caching, from measured token counts. Rates include the 272K long-context multipliers.

## 9. Stages, stops, and caps

Hard cap $350. Stage caps: pilot $15, chat benchmarks $90, coding $250 (`src/memstudy/budget.py`). The budget
guard refuses any call whose worst case would exceed a cap. Pilot: 30 LoCoMo questions (from 3
seeded conversations, to bound Mem0 ingestion), 20 LongMemEval-S questions, 5 coding tasks, all
arms. Stop and report if: the judge check fails; the coding memory-off solve rate is at or below
10% or at or above 90%; any usage field the cost accounting needs is missing from the API.

## 10. Known limits stated up front

Single seed. Single reader model and one vendor. Judge differs from each benchmark's default.
Mem0 OSS rather than Platform. Arm D not run. Reader snapshot not pinnable beyond the alias.
Proxy tokenizer for the fit check. Coding history is gold-patch experience, not trajectories.
