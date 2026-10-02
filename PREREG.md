# Pre-registration v1

Study: **Memory Layer or Full Context?** A controlled evaluation of long-term conversational
memory. Source: `Memstudy_Research_Proposal_Compact.docx` (the final proposal). Branch `memstudy`.
Written 2026-10-01, before any benchmark run. The settings are in `configs/study.yaml` and the
prompt texts in `src/memstudy/prompts.py`; their hash is written to every `run.json` as
`config_hash`. A change after tag `prereg-v1` needs a new tag.

## 1. Question, hypotheses, falsification

When the whole history fits in the model's context window, does a model still need an external
memory layer? No custom benchmark is introduced; two published benchmarks are used as published.

| ID | Hypothesis | Primary test | Falsified if |
| --- | --- | --- | --- |
| H1 | On LoCoMo (categories 1 to 4) and LongMemEval-S, the no-memory arm (A) is non-inferior to the memory system (Mem0, B) | Paired non-inferiority, 3 point margin, bootstrap CIs | Lower 95% bound of (A minus Mem0) falls below the margin, or Mem0 is significantly better |
| H2 | A cost crossover exists: beyond some queries per history a memory system is cheaper per question than cached full context | Cost model from measured token counts, with and without caching; crossover query count with bootstrap interval | No crossover in the tested range, or only beyond a realistic query count |
| H3 | Plain RAG (C) is within the margin of Mem0 | Paired non-inferiority, RAG versus Mem0 | Mem0 exceeds RAG by more than the margin |

The headline claim is falsified if a memory layer beats full context at history lengths that fit
in the window, or if no cost crossover appears. Either outcome is a reportable finding.

## 2. Task and data

Long-term conversational question answering: a multi-session history plus a question, and a short
free-text answer graded against a gold answer. Same definition for all arms.

- LoCoMo (`data/locomo/locomo10.json`): 10 conversations, 1,986 questions. Categories 1 to 4
  (1,540) are primary. The 446 adversarial questions are reported separately and graded for
  abstention.
- LongMemEval-S: 500 questions, each with its own haystack of about 104K tokens (measured), so
  histories are not shared and caching cannot amortize. 30 are abstention questions.
- Both used as published. Checksums in `configs/data_manifest.yaml`.
- Token census (`results/phase0/token_census.json`, o200k_base proxy): LoCoMo 10 to 21K tokens per
  history; LongMemEval-S 97K to 106K. Every history fits the 1.05M window with room to spare and
  none crosses the 272K long-context price threshold. Nothing is truncated: an item that does not
  fit is recorded as `does_not_fit`.

## 3. Reader and judge

| Role | Model | Settings |
| --- | --- | --- |
| Reader, all arms | `gpt-6-luna` (fallback GPT-5.6 Luna) | reasoning effort `none`, temperature 0, max output 512, one template |
| Judge, sole grader | `gpt-5-nano` | reasoning effort `minimal`, max output 256, one fixed prompt |

- Verified live on 2026-10-01 (`memstudy api-check`, $0.0003): with effort `none` the API accepts
  `temperature=0`; two identical calls returned the same answer; the cache breakpoint wrote and
  then read 2,125 tokens; billed cost matched the price formula exactly. The model page lists only
  the alias, so the served snapshot (`response.model` was `gpt-6-luna`) cannot be pinned further.
- Prices (`configs/prices.yaml`, sources and dates there): Luna $0.10 input, $0.01 cached, $0.125
  cache write, $0.50 output per 1M; above 272K input tokens 2x input and cache rates and 1.5x
  output for the whole request. Nano $0.05, $0.005, $0.40. Caching rules: minimum prefix 1,024
  tokens; write 1.25x; read 0.1x; lifetime at least 30 minutes, refreshed on reuse.
- Judge protocol: one fixed prompt (text in `prompts.py`, version v2, hashed into the config
  hash) with one rule for every question: the answer is correct if it has the core components of
  the gold answer and means the same thing. Dates and numbers match in any format. Where the gold
  says the information is not available, the answer must say so too. Where a question lists
  several accepted answers, any one suffices. Unparseable output is recorded as ungraded, never correct.
  Reasoning tokens are logged. No second grader and no human grading are used.
- Stability check: the judge is rerun on 100 answers (50 from arm A, 50 from arm B) and its flip
  rate is reported (`memstudy judge-flip`). On short-answer, non-abstention questions agreement
  with a deterministic containment test is reported as a diagnostic only
  (`memstudy judge-diagnostics`). Neither check gates the study.

## 4. Arms

All arms share the reader, template and questions; only how the history reaches the reader differs.
Arms B and C place retrieved items in the prompt under one shared token budget of 7,000 tokens
(Mem0's reported mean per query), filled with whole items in rank order from a pool of 200
candidates; nothing is cut mid-item.

- **A, no memory (baseline).** Full history first, question last. The prefix is byte-identical
  across questions on one history. An explicit cache breakpoint follows the history when the run
  asks at least two questions about it (LoCoMo, about 199 per history); for a single question
  (LongMemEval-S) there is no breakpoint, because a cache write with no later read is a 25%
  surcharge. Both modes are priced in the crossover analysis. LoCoMo questions are asked in
  sequence per conversation so the cached prefix is reused.
- **B, Mem0 (system under test).** `mem0ai==2.2.1` open source, one `user_id` per history
  (`<bench>_<history_id>`), local Qdrant store. Default extraction model `gpt-5-mini` and default
  embedder `text-embedding-3-small`, recorded here. `gpt-5-mini` is marked Deprecated on its
  model page; if it is withdrawn the study stops and the owner decides. Library defaults
  `threshold=0.1`, `rerank=false`. Ingestion: 10 turns per `add`, session date written into each
  message (the OSS `timestamp` argument is Platform-only). spaCy `en_core_web_sm` is required
  (without it Mem0 silently degrades to semantic-only retrieval, so the arm refuses to start).
  Telemetry off. The open-source library is used, not the hosted Platform, so numbers are not
  comparable to Mem0's published ones.
- **C, plain RAG (control for H3).** Verbatim chunks of whole turns, 256 tokens,
  `text-embedding-3-small` (the same embedder as Mem0), cosine ranking. Oversized turns are split,
  never cut.

Mem0's self-reported scores are never used as baselines. The prior numbers
in Pollertlam and Kornsuwannawit and Wolff and Bennati are reference points only.

## 5. Scoping (fairness)

- `user_id` is `"<bench>_<history_id>"`: one per LoCoMo conversation and per LongMemEval-S
  haystack, never shared across test cases.
- `tests/test_mem0_isolation.py` runs the real Mem0 class on a real local Qdrant store with an
  embedder giving every text the same vector (worst case for leakage) and shows a search under
  one id returns zero memories from another.

## 6. Statistics

- Accuracy per arm, benchmark and category; LoCoMo primary and adversarial reported separately.
- H1 and H3: paired non-inferiority on per-question outcomes, margin 3 percentage points (fixed
  here, before data). Differences are reported as A minus the comparison arm.
- Confidence intervals: 10,000 bootstrap resamples (seed 0). Both question-level and
  history-clustered intervals are reported side by side, and each conclusion states which it
  rests on. LoCoMo has only 10 histories, so its clustered intervals will be wide; for
  LongMemEval-S a history is one question.
- H2: cost per question from measured token counts and verified prices. Arm A pays one
  cache-writing call then cached reads; memory arms pay one-time ingestion plus per-question
  retrieval and reading. The crossover query count is where a memory arm's cumulative cost falls
  below arm A's, computed with and without caching (`src/memstudy/costmodel.py`), with a bootstrap
  interval.
- Secondary: input, cached, output and reasoning tokens, cost per question, crossover, latency
  (retrieval plus reader; ingestion time reported separately).

## 7. Cost and caps

Hard cap $350; stage caps pilot $15 and chat benchmarks $90, enforced by `budget.py` (refuses any
call whose worst case would exceed a cap). Every call is metered into `results/ledger.jsonl`;

Projection from the token census (reader and judge from the cost model, ingestion estimated):

| Item | LoCoMo | LongMemEval-S |
| --- | --- | --- |
| Arm A reader | about $0.4 cached | about $5.2 (no breakpoint) |
| Arm B ingestion plus reads | about $2 | about $24 |
| Arm C (plain RAG) | about $1.4 | about $1.4 |

## 8. Stops and limits

Stop and report if: the guard trips; `gpt-5-mini` or `temperature=0` is rejected; an API usage
field the accounting needs is missing; ingestion of any history fails or times out.

Stated limits: one reader, one seed, one grader whose error and family-level bias are not
independently measured (the reader and judge are both OpenAI models), only ten LoCoMo histories,
conversational benchmarks only, histories that fit in context by design, the open-source Mem0
rather than the hosted one, proxy tokenizer for the fit check, and results tied to the tested
versions.

## 9. Timeline

Per the proposal: implement the arms and pipeline (Oct 5 to 16); pilot on a small sample of each
benchmark to confirm caching and judge stability (Oct 19 to 23; 30 LoCoMo questions from 3
conversations, 20 LongMemEval-S questions); full LoCoMo (Oct 26 to Nov 6); full LongMemEval-S
(Nov 9 to 20); analysis (Nov 23 to 27); write-up (Nov 30 to Dec 11). Each paid stage needs the
owner's approval; the CLI refuses a paid command until tag `prereg-v1` exists and the matching
gate in `configs/approvals.yaml` is true.
