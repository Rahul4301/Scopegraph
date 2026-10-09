# Results

The reference record of every reported run: what was run, with which code, what came out, and
everything that had to change before the runs worked. Last updated 2026-10-09.

**Status.** Arms A (no memory) and C (plain RAG) are complete on all three benchmarks under prompts
v7. Arm B (Mem0) has not been run, so **H1 and H3 are not yet tested**: both are comparisons with
Mem0 on LongMemEval-S. Every comparison below is A vs C, which is not a registered test
(PREREG Section 6), and is reported as exploratory.

---

## 1. The six runs

All result files are in `results/`. Each one records its git commit, config hash, models and
every question. Superseded runs are in `data/old_runs/` and `data/old_smoke/` (Section 6).

| Arm | Benchmark | File | Questions answered | Cost | Wall time |
| --- | --- | --- | --- | --- | --- |
| A | LoCoMo | `no_memory_locomo_20261009_110627.json` | 1,980 / 1,986 | $1.28 | 92 min |
| A | LongMemEval-S | `no_memory_longmemeval_20261009_110634.json` | 500 / 500 | $5.25 | 38 min |
| A | MemoryAgentBench | `no_memory_memoryagentbench_20261009_132956.json` | 3,371 / 3,371 | $13.27 | 46 min |
| C | LoCoMo | `rag_locomo_20261009_144918.json` | 1,977 / 1,986 | $2.27 | 41 min |
| C | LongMemEval-S | `rag_longmemeval_20261009_144923_merged.json` | 500 / 500 | $1.59 | 52 min |
| C | MemoryAgentBench | `rag_memoryagentbench_20261009_151920_merged.json` | 3,371 / 3,371 | $3.12 | 37 min |

Total for the six runs: $26.78. Whole study to date: $69.41 ($68.81 full runs, $0.60 pilot and
smoke).

**Unanswered questions.** A LoCoMo has 6 and C LoCoMo has 9. In 14 of the 15 the judge (GPT-5
nano) ran out of its 2,048 output tokens (`IncompleteResponse`); the other was an OpenAI server
error that failed all four retries. The reader's answer is lost with the judge's. They are left
out of every comparison. They can be recovered with `--retry_failed` and `merge`.

**MemoryAgentBench counts.** 3,571 questions after excluding InfBench Sum (amendment A3); the 200
recsys questions are also excluded (the context is 1.48M tokens, larger than the window; PREREG
Section 2), leaving 3,371. The primary set is Accurate Retrieval plus Conflict Resolution without
the five LongMemEval-S copies: 2,500 questions.

### Provenance

A result file records the commit that was checked out when the run *finished*. Three runs were
still going when a later commit was made, so their recorded commit is later than the code they
ran. None of those later commits changes reading, grading or scoring for the run in question.

| File | Code it ran (commit at start) | Commit recorded | Config hash |
| --- | --- | --- | --- |
| A LoCoMo | `15a755e` | `9f6747a` | `894c0ef2199b801d` |
| A LongMemEval-S | `15a755e` | `15a755e` | `894c0ef2199b801d` |
| A MemoryAgentBench | `1ca6234` | `1ca6234` | `69831c61e30f45cf` |
| C LoCoMo | `cb145b5` | `a30ed30` | `69831c61e30f45cf` |
| C LongMemEval-S, first part (346) | `cb145b5` | `a30ed30` | `69831c61e30f45cf` |
| C LongMemEval-S, retry (154) | `b52cfd9` | `b52cfd9` | `69831c61e30f45cf` |
| C MemoryAgentBench, first part (3,263) | `a30ed30` | `b52cfd9` | `69831c61e30f45cf` |
| C MemoryAgentBench, retry (108) | `b52cfd9` | `b52cfd9` | `69831c61e30f45cf` |

The two config hashes differ in one setting only: the output limit of the InfBench summary
grader (8,192, then 32,768). InfBench is excluded, so the setting is never used. All six runs use
prompts v7, reader `gpt-6-luna` (effort none, temperature 0) and judge `gpt-5-nano` (effort
medium).

---

## 2. Results: no memory (A) vs plain RAG (C)

Only questions that both arms answered are compared. A − C is in percentage points; intervals are
95% bootstrap intervals that resample whole histories (the method of H1). Primary metrics: the
judge verdict on LoCoMo and LongMemEval-S, the benchmark's own pass test (`native_match`) on
MemoryAgentBench.

| Benchmark | Questions | A | C | A − C [95% CI] |
| --- | --- | --- | --- | --- |
| LoCoMo, categories 1 to 4 | 1,525 | 78.1% | 78.0% | +0.1 [−1.6, +1.6] |
| LongMemEval-S | 500 | 70.6% | 76.4% | **−5.8** [−9.4, −2.0] |
| MemoryAgentBench, primary set | 2,500 | 70.1% | 65.6% | **+4.5** [+2.6, +6.6] |

### LoCoMo (10 conversations)

| Category | Questions | A | C | A − C [95% CI] |
| --- | --- | --- | --- | --- |
| 1, multi-hop | 278 | 54.0 | 56.5 | −2.5 [−8.7, +2.8] |
| 2, temporal | 312 | 74.4 | 78.2 | −3.8 [−6.7, −0.6] |
| 3, open-domain | 95 | 51.6 | 53.7 | −2.1 [−8.9, +5.3] |
| 4, single-hop | 840 | 90.5 | 87.9 | +2.6 [+0.2, +4.7] |
| 5, adversarial (must abstain) | 446 | 46.6 | 41.0 | +5.6 [+0.7, +11.0] |

- Without the development conversation `conv-26` (PREREG Section 6), categories 1 to 4:
  A 78.5, C 78.6, A − C −0.1 [−2.0, +1.6], 1,376 questions.
- Mem0's own judge prompt (secondary, categories 1 to 4): A 87.4, C 86.6. It is more lenient
  than the study judge, and is reported only so these numbers can sit beside published Mem0 and
  Zep LoCoMo results.

### LongMemEval-S (500 histories, one question each)

| Type | Questions | A | C | A − C [95% CI] |
| --- | --- | --- | --- | --- |
| knowledge-update | 78 | 94.9 | 82.1 | +12.8 [+6.4, +20.5] |
| single-session-user | 70 | 97.1 | 98.6 | −1.4 [−5.7, +2.9] |
| single-session-assistant | 56 | 98.2 | 98.2 | 0.0 [−5.4, +5.4] |
| single-session-preference | 30 | 33.3 | 43.3 | −10.0 [−26.7, +6.7] |
| temporal-reasoning | 133 | 54.1 | 65.4 | −11.3 [−20.3, −2.3] |
| multi-session | 133 | 55.6 | 70.7 | −15.0 [−23.3, −6.8] |

### MemoryAgentBench

| Competency | Questions | Documents | A | C | A − C [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Accurate Retrieval (without LongMemEval-S copies) | 1,700 | 17 | 90.5 | 85.3 | +5.2 [+2.8, +7.9] |
| Conflict Resolution | 800 | 8 | 26.8 | 23.9 | +2.9 [+0.2, +6.1] |
| Long-Range Understanding (DetectiveQA only) | 71 | 10 | 81.7 | 77.5 | +4.2 [−4.2, +14.3] |
| Test-Time Learning (classification) | 500 | 5 | 90.2 | 85.2 | +5.0 [−0.2, +12.2] |
| LongMemEval-S copies (not primary) | 300 | 5 | 63.3 | 59.3 | +4.0 [+1.7, +6.3] |

Per source (one document each, so no interval): the largest gaps favour A on the longest
documents, `eventqa_full` (A 86.6, C 76.4) and `ruler_qa2_421K` (72.0, 63.0), and on
`icl_trec_fine` (81.0, 62.0). Multi-hop fact consolidation is near zero for both arms (1 to 5%
per source), so it separates neither.

### Cost per arm

| Benchmark | Questions per history | A | C |
| --- | --- | --- | --- |
| LoCoMo | about 200 | **$1.28** (cache hit 99.4%) | $2.27 |
| LongMemEval-S | 1 | $5.25 (no caching possible) | **$1.59** |
| MemoryAgentBench | about 75, documents about 200K tokens | $13.27 (cache hit 98.6%) | **$3.12** |

Mean seconds per question: A 4.4 to 5.4, C 3.6 to 4.8.

### Reading

1. **Where the answer has to be assembled from facts scattered over a long history, RAG beats
   full context**: LongMemEval-S multi-session (−15.0) and temporal reasoning (−11.3), LoCoMo
   temporal (−3.8). A focused 7,000-token context helps a reader at effort `none` more than
   100K+ tokens of history does. This is the main risk to H1, which compares A with Mem0 on the
   same benchmark.
2. **Where the newest fact or the exact text matters, full context beats RAG**: knowledge-update
   (+12.8; retrieval surfaces superseded facts), LoCoMo single-hop (+2.6) and abstention (+5.6),
   and almost all of MemoryAgentBench (long-document retrieval).
3. **Cost follows the shape H2 predicts**: with many questions per history cached full context is
   cheaper (LoCoMo); with one question per history RAG is about 3× cheaper (LongMemEval-S); with
   documents of about 200K tokens RAG is cheaper even with a 98.6% cache hit rate.

The reader settings were registered before these results and are not changed in response to
them.

---

## 3. Grader checks

PREREG Section 3 sets three checks on the judge. On these runs:

| Check | Threshold | Result |
| --- | --- | --- |
| Unparseable or cut-off judge output | at most 1% | 0% on LongMemEval-S; 0.30% (A) and 0.45% (C) on LoCoMo (judge cut-offs, counted as unanswered); 0.47% (A) and 0.21% (C) of MemoryAgentBench answers ungraded by the judge (the primary metric there is the pass test, which needs no judge) |
| Flip rate (judge rerun on 100 answers) | at most 5% | **not yet measured** (`memstudy judge-flip` needs the pilot) |
| Agreement with the containment test, short-answer questions | at least 85% | LoCoMo 83.3% (A), 84.6% (C); LongMemEval-S 82.5% (A), 81.9% (C); MemoryAgentBench 89.1% (A), 90.3% (C) |

LoCoMo and LongMemEval-S are below 85%, so by the registered rule containment accuracy is
reported beside the judge and a conclusion that differs between them is labelled
judge-dependent. On the same short-answer questions:

| Benchmark | Questions | Judge: A, C, A − C [95% CI] | Containment: A, C, A − C [95% CI] |
| --- | --- | --- | --- |
| LoCoMo, categories 1 to 4 | 1,051 | 80.8, 80.7, +0.1 [−1.7, +2.0] | 66.1, 66.7, −0.6 [−2.7, +1.3] |
| LongMemEval-S | 337 | 71.8, 80.1, −8.3 [−12.8, −3.9] | 63.8, 68.5, −4.7 [−8.3, −1.2] |

Neither conclusion is judge-dependent: LoCoMo is a tie and LongMemEval-S favours RAG under both.
Containment is lower than the judge in absolute terms because it misses paraphrases ("3" vs
"three children").

---

## 4. What changed before these runs, and why

The first full runs (arm A on LoCoMo and MemoryAgentBench, prompts v5, 2026-10-08) exposed
problems in the harness and in how each benchmark was graded. They were fixed in this order.
Every grading change is in PREREG.md as an amendment; the v5 runs are not study results.

### 4.1 Harness fixes from the v5 runs (commit `6822a1a`)

- **Prompt cache keys over 64 characters.** The cache key was `A-memoryagentbench_<history id>`;
  long ids broke OpenAI's 64-character limit and 1,400 MemoryAgentBench questions failed with
  HTTP 400 (not billed). Keys over the limit now keep a prefix plus a 16-character hash.
- **`--retry_failed FILE`** reruns only the questions an earlier result file did not answer,
  into a new file. **`--sources`** runs only some MemoryAgentBench sources. **`--task_prompts`**
  (exploratory, unregistered) wraps questions in MemoryAgentBench's verbatim templates.

### 4.2 Grading per benchmark: amendment A1, prompts v6

v5 used one reader template and one judge rule for every benchmark. Checking each benchmark's own
scorer, Mem0's evaluation code (`mem0ai/mem0` at `d85fcda`) and Zep's (`getzep/zep-papers`)
showed that the published numbers use per-benchmark grading:

- **LongMemEval-S** is now graded with LongMemEval's own judge prompt for each question type,
  copied verbatim from its `evaluate_qa.py` (temporal: off-by-one days allowed; knowledge-update:
  an answer that also gives the old value is correct; preference: the gold is a rubric;
  unanswerable questions: their own prompt). This is the change to a confirmatory measurement;
  it was made before any LongMemEval-S run.
- **LoCoMo** keeps the study judge, and adds Mem0's judge prompt (verbatim; Zep used the same) as
  a secondary grade for categories 1 to 4.
- **MemoryAgentBench** gets one answer-format line per source, adapted from the benchmark's own
  query templates (`prompts.TASK_INSTRUCTIONS`), identical for every arm; and its own pass test
  per source (`scoring.native_match`). The classification test now compares the parsed label: a
  substring test let gold `1` match `label: 17`. Detective QA compares the option letter.

The MemoryAgentBench part was made after the v5 arm A scores had been seen and is labelled so
in the amendment. Effect on arm A, v5 → v7 (amendment history only, not a finding):
classification 17.4% → 90.2%, conflict resolution 17.4% → 26.8%, primary set 66.4% → 70.1%.

All third-party grader text is in `src/memstudy/official_prompts.py`, unedited, and inside the
config hash.

### 4.3 Judge list rule and containment test: amendment A2, prompts v7

- The study judge required every listed gold item even when the question asked for one example
  ("What is a stress reliever for Evan?" against a gold listing five). Now all items are needed
  only when the question asks for all of them. Not used on LongMemEval-S, so descriptive
  measurements only.
- The containment diagnostic was raw substring matching and failed on format alone
  ("Kickboxing, Taekwondo" vs "kickboxing and taekwondo"). It now splits listed golds, ignores
  word order, stopwords, plurals and text after ";", and separates digits from letters
  (`pilot.gold_contained`). On the v5 LoCoMo run agreement rose from 70% to 84.5%; it was not
  tuned further against that data.

### 4.4 Exclusions: amendment A3 and PREREG Section 2

- **InfBench Sum excluded for every arm** (commit `ac93901`). Its summaries answer no registered
  question (H1 and H3 are on LongMemEval-S; one question per book gives H2 nothing to measure;
  RQ4 needs a short gold string), and they cost about 17M context tokens, three long grading
  calls each, and for Mem0 about $20 and 60 hours of ingestion. Decided after one book had been
  graded and four had hit the grader's output limit (which had been raised from 8,192 to 32,768
  in `9f6747a`). The MemoryAgentBench paper and later work do report InfBench Sum, so this is a
  stated limitation; Long-Range Understanding rests on DetectiveQA's 71 questions.
- **recsys excluded for every arm** (commit `cb145b5`). PREREG Section 2 already excluded the
  1.48M-token context. Arm A recorded it as `does_not_fit`, but RAG never sends the whole
  document and would have answered 200 questions that arm A could not.

Both exclusions are enforced in code (`execute.EXCLUDED_SOURCES`) for every run, retry and merge.

### 4.5 Making the RAG and Mem0 arms run to completion

The RAG runs failed three ways; each fix also applies to Mem0.

1. **Empty chunks** (commit `a30ed30`). The document chunker can emit a chunk of blank lines; the
   embeddings API rejects an empty input and the whole MemoryAgentBench run stopped. Blank chunks
   are now skipped before embedding or adding to Mem0. They carry no text, and the chunker itself
   stays lossless.
2. **One bad history stopped a whole run** (commit `a30ed30`). A failure while storing a history
   was not caught. It is now recorded as `ingest_failed` for each of that history's questions and
   the run moves on; `--retry_failed` picks them up.
3. **No retries on embedding and Mem0 calls** (commit `bfb3e1f`). Reader and judge calls retried
   transient errors; embedding calls and Mem0's internal calls did not, and one dropped
   connection (`SSL: bad record mac`) stopped the LongMemEval-S run. They now retry with the same
   policy (4 attempts, 1/2/4/8 s backoff).

Also:
- A stale RAG embedding cache for `conv-26` from an October 2 smoke test would have made that
  conversation's ingestion look free. Caches from the smoke test and from the crashed runs were
  moved out of `.cache/rag/` before the reported runs, so every history's ingestion cost and time
  is measured.
- **`memstudy merge`** (commit `b52cfd9`) combines a run and its retries into one file: each
  question keeps its first answered version, every case records the file it came from, and the
  run block lists every input's commit and config hash. Two reported files are merges:
  - C LongMemEval-S: 346 histories from the run that stopped on the dropped connection, plus 154
    from its retry.
  - C MemoryAgentBench: 3,263 questions, plus 108 retried (one eventqa document and one
    DetectiveQA document that failed to embed on a connection error before the retry fix, and
    two server errors).

### 4.6 Other corrections in PREREG.md

- The judge table said effort `minimal` and 256 output tokens; the registered config is effort
  `medium` and 2,048 (amendment A1 text).
- Only LongMemEval-S is confirmatory; LoCoMo and MemoryAgentBench are descriptive replications
  (correction in amendment A3).

---

## 5. Open items before the paper

1. **Run arm B (Mem0).** H1 and H3 both need Mem0 on LongMemEval-S. At the measured ingestion
   rate (14,160 tokens in 177 s, one history at a time) LongMemEval-S takes about 7.5 days and
   about $60. Options: run all 500 histories, register a random subset (and recompute the power
   curve), and/or ingest histories in parallel (each has its own store; Mem0's own benchmark
   suite ingests 10 at a time). LoCoMo takes about 1.2 hours and $2; MemoryAgentBench about 11
   to 26 hours and $8 to $11.
2. **Run the registered judge checks**: `pilot-select`, the pilot, `judge-flip`,
   `judge-diagnostics` (PREREG Section 3). The flip rate has never been measured.
3. **Recover the 15 unanswered LoCoMo questions** (`--retry_failed`, then `merge`), so both arms
   cover all 1,986.
4. **Development set.** `conv-26` is reported both ways above. The 20 LongMemEval-S pilot
   questions do not exist yet (`pilot-select` has not been run); when they are drawn they become
   development items and leave the reported set (PREREG Section 6).
5. **Judge validity** (not registered, recommended): hand-grade a stratified sample of about 200
   answers, and grade a sample with gpt-4o, the judge LongMemEval and MemoryAgentBench used.
6. Close the gates in `configs/approvals.yaml` when no run is going.

---

## 6. Superseded runs

Kept, not reported:

- `data/old_runs/`
  - the six v5 full runs of 2026-10-08 (arm A, LoCoMo and MemoryAgentBench, including the
    cache-key failure and the `--task_prompts` run);
  - the v7 arm A MemoryAgentBench run stopped for the summary-grader limit, its retry and their
    merge, replaced by one clean run;
  - the crashed C runs (MemoryAgentBench on the empty chunk, LongMemEval-S on the dropped
    connection), their retries, and an empty LongMemEval-S start;
  - the RAG embedding caches of the crashed runs (`rag_cache_*`).
- `data/old_smoke/`: smoke tests of 2026-10-02 (one LoCoMo conversation per arm) and the stale
  `conv-26` RAG cache.

`data/` is git-ignored; these files are kept locally.

---

## 7. Reproduce

Each command writes a new file in `results/` (never overwrites). Setup and data download are in
README.md.

```bash
.venv/bin/python -m memstudy run --memory_system no_memory --eval locomo --include_secondary
.venv/bin/python -m memstudy run --memory_system no_memory --eval longmemeval
.venv/bin/python -m memstudy run --memory_system no_memory --eval memoryagentbench --include_secondary --workers 8
.venv/bin/python -m memstudy run --memory_system rag --eval locomo --include_secondary --workers 8
.venv/bin/python -m memstudy run --memory_system rag --eval longmemeval
.venv/bin/python -m memstudy run --memory_system rag --eval memoryagentbench --include_secondary --workers 8
```

If a run stops early, finish it and combine the two files:

```bash
.venv/bin/python -m memstudy run <same flags> --retry_failed results/<stopped run>.json
.venv/bin/python -m memstudy merge results/<stopped run>.json results/<retry run>.json
```

The A vs C comparison tables in Section 2 are not yet a command; they were computed with a
paired, history-clustered bootstrap (4,000 resamples, seed 0) over the questions both arms
answered.
