# Pre-registration v1

Study: **When Does a Memory Layer Pay for Itself?** Accuracy and cost boundaries of agent memory
against cached full context. Source: [PROPOSAL.md](PROPOSAL.md), the proposal of record (it
replaces the earlier docx; arm labels follow this repository: A no memory, B Mem0, C plain RAG,
D Supermemory). Branch `memstudy`. Written 2026-10-01 and revised 2026-10-02, before any benchmark
run. The settings are in `configs/study.yaml` and the prompt texts in `src/memstudy/prompts.py`;
their hash is written to every `run.json` as `config_hash`. A change after tag `prereg-v1` needs a
new tag.

## 1. Question, hypotheses, falsification

When the whole history fits in the model's context window, does a model still need an external
memory layer? No custom benchmark is introduced; three published benchmarks are used as published
(LoCoMo, LongMemEval-S, MemoryAgentBench).

**Headline claim.** When the whole history fits in the context window, full-context prompting with
caching (arm A) is not less accurate than Mem0 (arm B) by more than 3 percentage points, so a
memory layer is not needed for accuracy. This is H1. Whether a memory layer is needed for cost is
a separate quantity, estimated in H2, and is not part of the falsification.

Each of H1 and H3 has three outcomes, decided on the interval for the difference (Section 6 gives
the interval level of each Holm step):

- **Non-inferior:** the lower limit is above -3 points.
- **Inferiority demonstrated (the hypothesis is falsified):** the upper limit is below -3 points.
- **Inconclusive:** otherwise, meaning the interval contains -3. An inconclusive result is never
  reported as evidence of inferiority or of non-inferiority.

A significant memory-layer advantage smaller than the margin does not falsify H1.

| ID | Hypothesis | Primary test | Outcomes |
| --- | --- | --- | --- |
| H1 | On LongMemEval-S, the no-memory arm (A) is non-inferior to the memory system (Mem0, B) | Paired non-inferiority, 3 point margin, history-clustered bootstrap interval | Non-inferior, inferiority demonstrated (falsified), or inconclusive, as above |
| H2 | Estimand, not a test: the number of queries per history beyond which Mem0 is cheaper per question than cached full context, as a function of history length and query spacing | Cost model from measured token counts, with and without caching; crossover query count with bootstrap interval | Not falsifiable. "No crossover" is a finding (full context is also cheaper), reported as no crossover up to the tested range |
| H3 | On LongMemEval-S, plain RAG (C) is non-inferior to Mem0 (B) | Paired non-inferiority, 3 point margin, same interval, for (C minus B) | Same three outcomes as H1 |

LoCoMo and MemoryAgentBench are pre-registered as descriptive replications of H1 and H3 (Section 6):
they have too few independent histories for a confirmatory test. Their results can support or
fail to support the direction found on LongMemEval-S but cannot establish non-inferiority. Arm D
(Supermemory, self-hosted) is analyzed with the same tests as arm B but is exploratory and sits
outside the confirmatory family, so the family and its power stay as registered.

H2 is a price-model extrapolation from measured token counts, cache hits and ingestion cost. A
crossover is measured directly only where one history gets many questions (LoCoMo, about 199;
MemoryAgentBench, 100 per context). LongMemEval-S asks one question per haystack, so no crossover
exists there and it is priced, not measured. The tested range is 1 to 1,000 queries per history;
"realistic" means at most 200, the largest per-history count in LoCoMo.

| H1 | H2 | Reading |
| --- | --- | --- |
| Non-inferior | No crossover up to 200 queries per history | Full context is enough: no accuracy or cost reason for a memory layer in this regime |
| Non-inferior | Crossover at Q* of at most 200 | A memory layer is a cost optimization above Q* queries per history, not an accuracy gain |
| Falsified (Mem0 more accurate by more than the margin: upper limit below -3) | Any | Headline falsified; report where and why |
| Inconclusive (the interval contains the margin, or the test is underpowered) | Any | No accuracy claim; report the interval and the power analysis |

## 2. Task and data

Long-term conversational question answering: a multi-session history plus a question, and a short
free-text answer graded against a gold answer. Same definition for all arms.

- LoCoMo (`data/locomo/locomo10.json`): 10 conversations, 1,986 questions. Categories 1 to 4
  (1,540) are primary. The 446 adversarial questions are reported separately and graded for
  abstention.
- LongMemEval-S: 500 questions, each with its own haystack of 97K to 106K tokens (measured with
  the proxy tokenizer), so histories are not shared and caching cannot amortize. 30 are
  abstention questions.
- MemoryAgentBench (`ai-hyz/MemoryAgentBench`, pinned revision `7ea06698`, four parquet files in
  `data/memoryagentbench/`): 146 rows, each one long context shared by many questions, 3,671
  questions in all. Accurate retrieval and conflict resolution (selective forgetting) are the
  primary descriptive competencies (30 contexts, 2,800 questions before the exclusion below);
  test-time learning and long-range understanding are exploratory. No MemoryAgentBench result is
  confirmatory. 145 of 146 contexts fit the window (token census in
  `results/phase0/mab_token_census.json`); `recsys_redial_full` (1.48M tokens) is excluded and
  reported. Five accurate-retrieval rows (`longmemeval_s*`, 300 questions) repeat LongMemEval-S
  content and are excluded from the MemoryAgentBench primary set to avoid double counting,
  leaving 25 contexts and 2,500 questions; the excluded rows are reported separately. Contexts
  are single documents, not sessions: arm A sends the text
  verbatim, arm B adds it in 4,096-token chunks, arm C indexes it in 256-token verbatim chunks.
  Scoring: substring match against the accepted answers (the benchmark's own style of metric,
  implemented in `scoring.py`, equivalence to the original code not yet checked) is primary;
  the judge verdict is secondary. The benchmark's own task prompts are not reproduced: the same
  reader template is used for every item and arm, which is a stated limitation.
- All three benchmarks are used as published. Checksums in `configs/data_manifest.yaml`.
- Composite long histories (Section 10) are built from LongMemEval-S sessions and are exploratory.
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
  (`<bench>_<history_id>`), local Qdrant store. Extraction model `gpt-6-luna`, the same model as
  the reader, with effort `none` sent explicitly (Mem0 does not know Luna and would otherwise
  send parameters Luna rejects). No other chat model is used anywhere: the models are GPT-6 Luna
  (reader and extraction) and GPT-5 nano (judge and fallback). Embedder `text-embedding-3-small`, the library default.
  Pre-registered fallback: if Luna is withdrawn or returns model-not-found, extraction switches to
  `gpt-5-nano` (priced and verified in `configs/prices.yaml`, and the extraction model used by
  Pollertlam and Kornsuwannawit), recorded as `extraction_model_fallback` in `configs/study.yaml`.
  Arm B is then re-run in full for any benchmark that already used Luna; extraction models are
  never mixed within a benchmark. The fallback has not yet been tested with Mem0 `2.2.1`. Library
  defaults
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

- **D, Supermemory, self-hosted (exploratory).** The self-hosted server binary (release
  `server-v0.0.8`; the repository `github.com/supermemoryai/supermemory` is MIT licensed but, as
  checked 2026-10-02, holds no server or extraction source: `apps/` has docs, mcp, web and
  playgrounds, `packages/` has SDKs and tools, so the extraction prompt cannot be audited), one container per history (`container_tag` = the same
  `<bench>_<history_id>` scope as arm B), `task_type` memory, `search_mode` hybrid (the server's
  documented default: extracted memories plus raw document chunks in one ranked list, so arm D
  is not a pure extraction system and partly overlaps with arm C; this is stated wherever its
  results appear). Chosen after one development case (`conv-26`): memories-only mode returned
  about 1,700 context tokens and 63.2% judge accuracy, hybrid about 2,700 tokens and 68.4%, a
  difference inside the noise of 152 questions, so the choice rests on hybrid being the
  product's own default, not on that result. Rerank and query rewriting are off. The server is configured with the same extraction model
  as arm B (`OPENAI_MODEL`) and the same embedder (provider `openai`, `text-embedding-3-small`,
  1,536 dimensions). Ingestion is asynchronous on the server, so the arm polls until every
  document is done before any query. It is reported as "Supermemory (self-hosted, shared
  extraction model)" and never as the hosted product, whose extraction models are proprietary.
  Known gaps, to be settled before arm D is run: neither the documentation nor the public
  repository says whether extraction reads only user turns (checked on a smoke run by reading
  the stored memories, and covered by the single-session-assistant sensitivity analysis); the server's own model calls are
  made with its own key and are not metered, so arm D spend is estimated from our token counts at
  the shared models' verified prices (input tokens only, a lower bound) and marked estimated in
  the ledger; the server's search `threshold` defaults to 0.6, which is tuned to its local embedder: with
  `text-embedding-3-small` the best similarity on a smoke question was 0.55 and the server
  returned nothing. Arm D therefore uses 0.1, Mem0's library default, so neither arm is tuned
  against the other (the smoke result at 0.6 is a development finding, not a result).
  The server rejects search limits above 100, so arm D's candidate pool is 100 where arms B and C
  use 200; the shared 7,000-token budget is unchanged and extracted facts are short, so the pool,
  not the budget, is the binding limit for D (reported, not tuned).
  Gold-in-store for arm D comes from the server's memory-list endpoint (`POST /v4/memories/list`,
  latest and not-forgotten versions), the same verbatim test as arm B.

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
  coincide. The primary test is H1. Holm's procedure adjusts across the two tests (levels below).
- **Interval:** paired per-question differences, 10,000 bootstrap resamples of histories (seed 0),
  two-sided intervals. Non-inferiority is claimed only if its lower limit is above -3 points.
- **Holm levels (family alpha 0.05, two tests).** Step 1: the test with the more favorable
  result (the larger lower limit; H1 if they tie) is judged on a 97.5% two-sided interval
  (alpha 0.025 per test). Only if it clears the margin is step 2 run: the other test is judged on
  the 95% two-sided interval (alpha 0.05). If step 1 fails, neither non-inferiority claim is
  made. Both intervals are reported for both tests whatever the outcome.
  The history-clustered interval is the primary one for every benchmark; the question-level
  interval is reported only as a sensitivity check.
- **Descriptive, not confirmatory:** LoCoMo (10 histories) and MemoryAgentBench (25 primary
  contexts after the overlap exclusion). With so few clusters a percentile bootstrap undercovers,
  so their intervals are reported but no non-inferiority claim rests on them; conclusions on
  these are worded as consistent or not with the LongMemEval-S result.
- **Margin:** 3 points was chosen before any data as the largest accuracy loss accepted in
  exchange for dropping a memory layer. It is not derived from data, and it is kept at 3 points
  on purpose: changing it after seeing the pilot discordance would make it data-driven. Prior
  work reports gaps between memory and full context of tens of points (Pollertlam and
  Kornsuwannawit: 57.68% against 92.85% on LoCoMo), so 3 points is tight enough that a
  non-inferiority claim means something. It is demanding at this sample size, which the power
  curve below states. Whether each conclusion also holds at 2 and at 5 points is reported as
  exploratory.
- **Sample.** The pilot's 20 LongMemEval-S questions are development items (below), so the
  confirmatory sample is 480 questions. Each history still holds one question.
- **Power curve (computed before the full run).** With d the discordance rate (the share of
  questions where exactly one of two arms is right), n = 480 and a true difference of zero, the
  expected lower limit is -z * sqrt(d / 480) and the power is the probability that it exceeds
  -3 points. Using z = 1.960 (95%) and z = 2.241 (97.5%, Holm step 1):

  | d | Expected lower limit, 95% | Power, 95% | Expected lower limit, 97.5% | Power, 97.5% |
  | --- | --- | --- | --- | --- |
  | 5% | -2.00 | 84% | -2.29 | 76% |
  | 8% | -2.53 | 64% | -2.89 | 53% |
  | 10% | -2.83 | 55% | -3.23 | 44% |
  | 12% | -3.10 | 48% | -3.54 | 37% |
  | 15% | -3.46 | 40% | -3.96 | 29% |
  | 20% | -4.00 | 31% | -4.57 | 22% |
  | 25% | -4.47 | 26% | -5.11 | 18% |
  | 30% | -4.90 | 22% | -5.60 | 15% |

  The expected lower limit stays above -3 points only when d is at most about 11% (95%) or
  about 9% (97.5%). At a true difference of zero the power never reaches 80% for d above 5%.
  A true advantage for arm A relaxes this. The curve is recomputed from the pilot's measured
  discordance before the full run and the result reported. A non-significant result in an
  underpowered test is labelled inconclusive, never presented as evidence against
  non-inferiority; a result that clears the margin stands either way.
- **Sensitivity analyses (exploratory).** (1) The same tests with the single-session-assistant
  category excluded (53 of the 480 questions): Mem0's default extraction reads only user
  messages, so assistant-only facts are not extracted, and this shows whether the conclusion
  depends on that. (2) The same tests at margins of 2 and 5 points.
- **Development set.** Items used while building the harness and the judge are not reported:
  the LoCoMo conversation used in smoke runs (`conv-26`, all questions), the pilot sample from
  `memstudy pilot-select` (30 LoCoMo questions from `conv-26`, `conv-47`, `conv-50`; 20
  LongMemEval-S questions), and any item answered in a run with stage smoke or pilot. Reported
  results come from full (stage chat) runs only, and these items are dropped from them at
  analysis time (173 of the 1,540 primary LoCoMo questions, leaving 1,367; 20 of the 500
  LongMemEval-S questions, leaving 480). The judge prompt and effort were changed during
  development (`prompts.py` version v2, effort `medium`), which is why these items are not clean.
  The composite histories exclude the 20 LongMemEval-S development questions by construction.
  Existing smoke result files under `results/` are kept unchanged and are not study results.
- All other comparisons (categories, exploratory MemoryAgentBench competencies, sensitivity
  analyses, judge-stability checks, arm D, and everything in Section 10) are exploratory and
  reported descriptively without adjustment.
- H2: cost per question from measured token counts and verified prices. Arm A pays one
  cache-writing call then cached reads; memory arms pay one-time ingestion plus per-question
  retrieval and reading. The crossover query count is where a memory arm's cumulative cost falls
  below arm A's, computed with and without caching (`src/memstudy/costmodel.py`), with a bootstrap
  interval over histories. Query spacing is a variable: a query that arrives within the cache
  lifetime (30 minutes, refreshed on reuse) of the previous one finds the prefix warm, otherwise
  arm A pays a cache write again. The warm fraction is measured from logged cached-token counts
  and simulated for fixed, burst and Poisson arrival schedules (Section 10).
- Secondary: input, cached, output and reasoning tokens, cost per question, crossover, latency
  (retrieval plus reader; ingestion time reported separately).

## 7. Cost and caps

Hard cap $350; stage caps pilot $15 and chat benchmarks $90, enforced by `budget.py` (refuses any
call whose worst case would exceed a cap). Every call is metered into `results/ledger.jsonl`.

Projection from the token census (reader and judge from the cost model, ingestion estimated):

| Item | LoCoMo | LongMemEval-S |
| --- | --- | --- |
| Arm A reader | about $0.4 cached | about $5.2 (no breakpoint) |
| Arm B ingestion plus reads | about $2 | about $24 |
| Arm C (plain RAG) | about $1.4 | about $1.4 |

The core design comes to about $35 under current price cards, against the $350 cap (the arm B
line above predates the switch of extraction to Luna and is re-run from the pilot). Extensions
(Section 10) are costed from the pilot; the arm A reader cost of the composite histories is about
$3 (41 questions per history, cached, 272K surcharge included), and arm D ingestion is an
estimate (Section 4).

MemoryAgentBench is not in this table. Its 145 fitting contexts hold about 26M tokens, and arm A
re-reads each context once per question (about 0.7B input tokens in all), so caching is required
and 23 contexts also cross the 272K price threshold. A dollar projection comes from the pilot; the
guard stops the run at the cap either way.

## 8. Stops and limits

Stop and report if: the guard trips; `temperature=0` is rejected; an API usage field the
accounting needs is missing; ingestion of any history fails or times out; the judge's unparseable
rate exceeds 1% in the pilot. A withdrawn extraction model is handled by the fallback in Section 4, not
a stop.

Stated limits: one reader, one seed, one grader whose error and family-level bias are not
independently measured (the reader and judge are both OpenAI models), only ten LoCoMo histories,
conversational benchmarks only, histories that fit in context by design, the open-source Mem0
rather than the hosted one, proxy tokenizer for the fit check, results tied to the tested
versions, MemoryAgentBench run with one generic reader template rather than its own task prompts,
Supermemory run self-hosted with a shared extraction model so it does not describe the hosted
product, composite histories that are synthetic, and public benchmarks that the reader model may
have seen in training.

## 9. Timeline

Ten weeks, fitted to a December 2026 graduation. Each paid stage needs the owner's approval; the
CLI refuses a paid command until tag `prereg-v1` exists and the matching gate in
`configs/approvals.yaml` is true.

| Week | Dates (2026) | Work |
| --- | --- | --- |
| 1 to 2 | Oct 5 to Oct 16 | Finish arms and the logging and grading pipeline; gold-in-store logging; composite-history constructor; decide which extensions enter the registration; create the registration tag |
| 3 | Oct 19 to Oct 23 | Pilot (30 LoCoMo questions from 3 conversations, 20 LongMemEval-S questions); judge checks; power curve; cost estimates for each extension; go or no-go per extension |
| 4 to 5 | Oct 26 to Nov 6 | Full LoCoMo, all registered arms; as-of queries and write granularity |
| 6 to 7 | Nov 9 to Nov 20 | Full LongMemEval-S, all registered arms |
| 8 | Nov 23 to Nov 27 | Length-axis runs. MemoryAgentBench on the primary descriptive competencies only if schedule and budget allow; it is dropped before the length axis |
| 9 | Nov 30 to Dec 4 | Analysis: tests, intervals, crossover surfaces, stage attribution |
| 10 | Dec 7 to Dec 11 | Write-up and buffer |

## 10. Extensions (exploratory)

These are registered so that their settings are fixed before any data. None is part of the
confirmatory family (H1 and H3); all are reported with intervals and without adjustment. A
subsection that the owner drops before tag `prereg-v1` is removed with its code left unused; an
extension added after the tag is exploratory by definition and needs a new tag if it changes a
registered setting. Paid runs are gated as in Section 9; the composite benchmark uses the gate
`stage_chat_composite`, and arm D uses `g2_supermemory_self_hosted` (a gate that is absent from
`configs/approvals.yaml` counts as closed).

**10.1 Stage attribution (RQ4).** For each question answered by a memory arm (B, and D) the
per-case record stores `gold_in_store`: an accepted answer appears, after normalization, in the
text of everything the system holds for the history. It is `None` for arms without a store and for
abstention questions. Together with the existing `gold_in_context` and the judge verdict it gives
the attribution in each run's summary: not stored (extraction), stored but not in the reader's
context (retrieval), in context but graded wrong (reading). Extracted facts paraphrase the
source, so both tests are lower bounds, and a judge-based check on a fixed random sample of 200
questions (seed 0) estimates the bias before the attribution is interpreted. For arm A the
analysis reports reader accuracy by evidence position and history length instead.

**10.2 Query spacing (H2, RQ2).** `costmodel.py` simulates arrival schedules against the cache
lifetime (30 minutes, refreshed on reuse): a query arriving within the lifetime of the previous
one reads the cached history, otherwise arm A writes it again. Schedules: fixed gaps of 1, 10, 30
and 60 minutes; bursts of 5 queries 10 seconds apart separated by 2 hours; and Poisson arrivals
with a mean gap of 10 minutes (seed 0). The crossover query count is computed for each schedule
with and without the cache breakpoint over the tested range of 1 to 1,000 queries, and the warm
fraction is also measured directly from the cached-token counts logged on arm A calls.

**10.3 Composite long histories (RQ3).** Built by `loaders/composite.py` from LongMemEval-S, seed 0:
- Target lengths 200K, 400K, 600K, 800K and 950K tokens (proxy tokenizer). The usable window is
  1.05M tokens less the 5% safety margin, so the largest target leaves about 47K tokens for the
  question and the output. Arm A uses the usual fit check, and the 272K price surcharge applies
  above that length.
- Questions: eligible questions are those that are not abstention and not development items,
  sorted by id, shuffled with the seed, and the first 40 kept. The subset size is fixed from the
  pilot variance and the budget cap before the run and may be lowered, never raised, after the
  tag. The same questions are used at every length, so accuracy at different lengths is
  measured on the same items. At the shipped settings 41 questions are asked at every length,
  because one further question's evidence lies in the same sessions.
- Distractors: one seeded shuffle of the sessions that are evidence for no question in
  LongMemEval-S, so a distractor is never another question's gold evidence. Conflict screen: a
  distractor is also rejected when it contains at least max(3, ceil(0.5 * k)) of a question's k
  content words (lowercase alphanumeric tokens of 4 or more characters, minus a fixed stop list).
  The build report lists, for each composite, the session counts and the number screened out (5
  to 276 at the shipped settings).
- Sessions are ordered by timestamp, ties in original order, and each item records the relative
  position of its evidence sessions for the position analysis. Composite histories are synthetic
  and are reported as such.

**10.4 As-of queries (LoCoMo).** `run --asof`. Each question is asked after the latest session
cited in its evidence (`D<n>:<m>`, counting only sessions that exist) and again on the full
conversation; a question whose checkpoint is the last session, or whose evidence cannot be read,
is asked once, at the end. A checkpoint history is the conversation cut after that session, and
all checkpoints of one conversation share one memory store, so memory arms ingest sessions
incrementally and the store at each checkpoint holds only past sessions. Questions run in order
of conversation and checkpoint. Arm A receives the history up to the checkpoint; its transcript
is a prefix of the next checkpoint's, every checkpoint history carries the cache breakpoint, and
cache behavior under growing prefixes is measured, not assumed. Accuracy and cumulative cost are
reported against the number of sessions seen. Result files are named
`<system>_locomo_asof_<time>.json`.

**10.5 Write granularity (arm B).** `run --memory_system mem0 --write_granularity turn|ten|session`
sets the messages per ingestion call: one turn, ten turns (the registered default, used in every
confirmatory and core run), or one whole session. The setting is part of the config hash and is
appended to the result file name. Each setting is compared with arm A on LoCoMo.

**10.6 Arm D, Supermemory self-hosted.** Defined in Section 4; exploratory; same tests as arm B,
outside the Holm family.
