# Pre-registration v1

Study: **Memory Layer or Full Context?** A controlled evaluation of long-term conversational
memory. Source: `Memstudy_Research_Proposal_Compact.docx` (the final proposal). Branch `memstudy`.
Written 2026-10-01, before any benchmark run. The settings are in `configs/study.yaml` and the
prompt texts in `src/memstudy/prompts.py`; their hash is written to every `run.json` as
`config_hash`. A change after tag `prereg-v1` needs a new tag.

## 1. Question, hypotheses, falsification

When the whole history fits in the model's context window, does a model still need an external
memory layer? No custom benchmark is introduced; three published benchmarks are used as published
(LoCoMo, LongMemEval-S, MemoryAgentBench).

**Headline claim.** When the whole history fits in the context window, full-context prompting with
caching (arm A) is not less accurate than Mem0 (arm B) by more than 3 percentage points, so a
memory layer is not needed for accuracy. This is H1. Whether a memory layer is needed for cost is
a separate quantity, estimated in H2, and is not part of the falsification.

| ID | Hypothesis | Primary test | Falsified if |
| --- | --- | --- | --- |
| H1 | On LongMemEval-S, the no-memory arm (A) is non-inferior to the memory system (Mem0, B) | Paired non-inferiority, 3 point margin, history-clustered bootstrap interval | The lower limit of the 95% interval for (A minus B) is below -3 points. A significant Mem0 advantage smaller than the margin does not falsify H1 |
| H2 | Estimand, not a test: the number of queries per history beyond which Mem0 is cheaper per question than cached full context | Cost model from measured token counts, with and without caching; crossover query count with bootstrap interval | Not falsifiable. "No crossover" is a finding (full context is also cheaper), reported as no crossover up to the tested range |
| H3 | On LongMemEval-S, plain RAG (C) is non-inferior to Mem0 (B) | Paired non-inferiority, 3 point margin, same interval | The lower limit of the 95% interval for (C minus B) is below -3 points |

LoCoMo and MemoryAgentBench are pre-registered as descriptive replications of H1 and H3 (Section 6):
they have too few independent histories for a confirmatory test. Their results can support or
fail to support the direction found on LongMemEval-S but cannot establish non-inferiority.

H2 is a price-model extrapolation from measured token counts, cache hits and ingestion cost. A
crossover is measured directly only where one history gets many questions (LoCoMo, about 199;
MemoryAgentBench, 100 per context). LongMemEval-S asks one question per haystack, so no crossover
exists there and it is priced, not measured. The tested range is 1 to 1,000 queries per history;
"realistic" means at most 200, the largest per-history count in LoCoMo.

| H1 | H2 | Reading |
| --- | --- | --- |
| Holds | No crossover up to 200 queries per history | Full context is enough: no accuracy or cost reason for a memory layer in this regime |
| Holds | Crossover at Q* of at most 200 | A memory layer is a cost optimization above Q* queries per history, not an accuracy gain |
| Fails (Mem0 better by more than the margin) | Any | Headline falsified; report where and why |
| Inconclusive (interval spans the margin, or underpowered) | Any | No accuracy claim; report the interval |

## 2. Task and data

Long-term conversational question answering: a multi-session history plus a question, and a short
free-text answer graded against a gold answer. Same definition for all arms.

- LoCoMo (`data/locomo/locomo10.json`): 10 conversations, 1,986 questions. Categories 1 to 4
  (1,540) are primary. The 446 adversarial questions are reported separately and graded for
  abstention.
- LongMemEval-S: 500 questions, each with its own haystack of about 104K tokens (measured), so
  histories are not shared and caching cannot amortize. 30 are abstention questions.
- MemoryAgentBench (`ai-hyz/MemoryAgentBench`, pinned revision `7ea06698`, four parquet files in
  `data/memoryagentbench/`): 146 rows, each one long context shared by many questions, 3,671
  questions in all. Accurate retrieval and conflict resolution (selective forgetting) are the
  confirmatory competencies (30 contexts, 2,800 questions before the exclusion below); test-time learning and long-range
  understanding are exploratory. 145 of 146 contexts fit the window (token census in
  `results/phase0/mab_token_census.json`); `recsys_redial_full` (1.48M tokens) is excluded and
  reported. Five accurate-retrieval rows (`longmemeval_s*`, 300 questions) repeat LongMemEval-S
  content and are excluded from the MemoryAgentBench confirmatory set to avoid double counting,
  leaving 25 contexts and 2,500 questions; the excluded rows are reported separately. Contexts are single documents, not sessions: arm A sends the text
  verbatim, arm B adds it in 4,096-token chunks, arm C indexes it in 256-token verbatim chunks.
  Scoring: substring match against the accepted answers (the benchmark's own style of metric,
  implemented in `scoring.py`, equivalence to the original code not yet checked) is primary;
  the judge verdict is secondary. The benchmark's own task prompts are not reproduced: the same
  reader template is used for every item and arm, which is a stated limitation.
- Both LoCoMo and LongMemEval-S used as published. Checksums in `configs/data_manifest.yaml`.
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
- Judge checks run in the pilot, before any full run, with thresholds fixed here:
  - Unparseable output must be at most 1% of judged answers. Above that the study stops and the
    owner decides, because those answers would be silently dropped.
  - Flip rate: the judge is rerun on 100 answers (50 from arm A, 50 from arm B)
    (`memstudy judge-flip`). It must be at most 5%. Above that, every answer in the study is
    graded by the majority of three nano runs (about three times the small judge cost) and the
    flip rate is reported with every accuracy figure.
  - Agreement with a deterministic containment test on short-answer, non-abstention questions
    (`memstudy judge-diagnostics`) must be at least 85%. Below that, containment accuracy is
    reported beside every judge accuracy and a conclusion that differs between the two is labelled
    judge-dependent.

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
  model page. Pre-registered fallback: if it is withdrawn or returns model-not-found, extraction
  switches to `gpt-5-nano` (priced and verified in `configs/prices.yaml`, and the extraction model
  used by Pollertlam and Kornsuwannawit), recorded as `extraction_model_fallback` in
  `configs/study.yaml`. Arm B is then re-run in full for any benchmark that already used the
  default; extraction models are never mixed within a benchmark. The fallback has not yet been
  tested with Mem0 `2.2.1`. Library defaults
  `threshold=0.1`, `rerank=false`. Ingestion: 10 turns per `add`, session date written into each
  message (the OSS `timestamp` argument is Platform-only). Mem0's default extraction prompt reads
  only user messages, so LoCoMo's two peer speakers are both sent as user messages prefixed with
  their names; LongMemEval keeps its real user and assistant roles, so assistant-only facts are
  not extracted (Mem0 as released). spaCy `en_core_web_sm` is required
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
  Differences are reported as A minus the comparison arm (and C minus B for H3).
- **Confirmatory tests: H1 and H3 on LongMemEval-S only.** Each of its 500 histories holds one
  question, so histories are independent and the question-level and history-clustered intervals
  coincide. The primary test is H1. Holm's procedure adjusts across the two tests.
- **Interval:** paired per-question differences, 10,000 bootstrap resamples of histories (seed 0),
  two-sided 95% interval. Non-inferiority is claimed only if its lower limit is above -3 points.
  The history-clustered interval is the primary one for every benchmark; the question-level
  interval is reported only as a sensitivity check.
- **Descriptive, not confirmatory:** LoCoMo (10 histories) and MemoryAgentBench (25 confirmatory
  contexts after the overlap exclusion). With so few clusters a percentile bootstrap undercovers,
  so their intervals are reported but no non-inferiority claim rests on them; conclusions on
  these are worded as consistent or not with the LongMemEval-S result.
- **Margin:** 3 points was chosen before any data as the largest accuracy loss accepted in
  exchange for dropping a memory layer. It is not derived from data. Whether each conclusion
  also holds at 2 and at 5 points is reported as exploratory.
- **Power:** from pilot discordance (the share of questions where exactly one of two arms is
  right) the power at a true difference of zero is computed for n = 500. If it is below 80%, a
  non-significant result is labelled inconclusive, never presented as evidence against
  non-inferiority; a result that clears the margin stands either way.
- All other comparisons (categories, exploratory MemoryAgentBench competencies, sensitivity
  margins, judge-stability checks) are exploratory and reported descriptively without adjustment.
- H2: cost per question from measured token counts and verified prices. Arm A pays one
  cache-writing call then cached reads; memory arms pay one-time ingestion plus per-question
  retrieval and reading. The crossover query count is where a memory arm's cumulative cost falls
  below arm A's, computed with and without caching (`src/memstudy/costmodel.py`), with a bootstrap
  interval over histories.
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

MemoryAgentBench is not in this table. Its 145 fitting contexts hold about 26M tokens, and arm A
re-reads each context once per question (about 0.7B input tokens in all), so caching is required
and 23 contexts also cross the 272K price threshold. A dollar projection comes from the pilot; the
guard stops the run at the cap either way.

## 8. Stops and limits

Stop and report if: the guard trips; `temperature=0` is rejected; an API usage field the
accounting needs is missing; ingestion of any history fails or times out; the judge's unparseable
rate exceeds 1% in the pilot. A withdrawn `gpt-5-mini` is handled by the fallback in Section 4, not
a stop.

Stated limits: one reader, one seed, one grader whose error and family-level bias are not
independently measured (the reader and judge are both OpenAI models), only ten LoCoMo histories,
conversational benchmarks only, histories that fit in context by design, the open-source Mem0
rather than the hosted one, proxy tokenizer for the fit check, results tied to the tested
versions, MemoryAgentBench run with one generic reader template rather than its own task prompts,
and public benchmarks that the reader model may have seen in training.

## 9. Timeline

Per the proposal: implement the arms and pipeline (Oct 5 to 16); pilot on a small sample of each
benchmark to confirm caching and judge stability (Oct 19 to 23; 30 LoCoMo questions from 3
conversations, 20 LongMemEval-S questions); full LoCoMo (Oct 26 to Nov 6); full LongMemEval-S
(Nov 9 to 20); MemoryAgentBench, confirmatory competencies first (Nov 23 to 27); analysis (Nov 30 to Dec 4); write-up (Dec 7 to Dec 11). Each paid stage needs the
owner's approval; the CLI refuses a paid command until tag `prereg-v1` exists and the matching
gate in `configs/approvals.yaml` is true.
