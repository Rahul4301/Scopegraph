# MemoryAgentBench run notes (arm A, no memory), 2026-10-08

## What happened

| # | Issue | Cause | Fix |
|---|---|---|---|
| 1 | 1,400 of 3,671 questions failed with HTTP 400 | `prompt_cache_key` was `A-memoryagentbench_<history_id>`; long ids exceeded OpenAI's 64-character limit. Requests were rejected before billing. | `runner.cache_key`: keys over 64 chars keep a prefix plus a 16-char hash. Short keys are unchanged. |
| 2 | 100 Long_Range_Understanding questions errored | `infbench_sum` asks for a 1,000-1,200 word summary; reader output is capped at `max_output_tokens: 512` (registered). All 100 stopped at 512 tokens. $2.31 charged. | Not fixed. **Excluded** and reported. The same cap applies to every arm. |
| 3 | 200 questions skipped | `recsys_redial_full` is 1.48M tokens, larger than the 1.05M window. | Expected. Never truncated. |
| 4 | Classification and fact consolidation scored very low | The registered generic reader prompt never states these tasks. The benchmark puts the instructions ("output `label: N`", "newer fact has larger serial number") in its own templates (PREREG §5 limitation). | Added an opt-in `--task_prompts` sensitivity run (below). |

## Code added

- `--retry_failed FILE`: rerun only questions that errored or were not run.
- `--sources PREFIXES`: run only the listed MemoryAgentBench sub-datasets.
- `--task_prompts`: wrap fact-consolidation and classification questions in MemoryAgentBench's own templates, copied verbatim from its `utils/templates.py`. **Exploratory, not registered.** Changes the config hash; result files are named `*_taskprompts_*`.

## Results (arm A, `results/no_memory_memoryagentbench_20261008_191950.json`)

3,371 of 3,671 answered. Substring match is the primary metric.

| Split | n | Score |
|---|---|---|
| **Confirmatory (AR + CR)** | 2,500 | **66.4%** |
| Accurate_Retrieval | 2,000 | 81.8% |
| Conflict_Resolution | 800 | 17.4% (single-hop 32.2%, multi-hop 2.5%) |
| Long_Range_Understanding (Detective QA only) | 71 | 71.8% |
| Test_Time_Learning (classification) | 500 | 17.4%\* |

\* Scored by whole-number match. The judge gave 40 false positives here (fluent answers with no label), and substring match gave 6 (for example, gold `4` matched inside `343`).

**Why these splits are low.** Measured from the answers:

- **Classification**: 59% of answers addressed the question instead of giving a label. 22% said "not mentioned". When the model did give a label, it was right 94% of the time.
- **Single-hop**: 67% of answers were the older, overridden fact. The updates are deliberately counterfactual (for example, "Nat King Cole plays J-pop"), so the model trusts real-world knowledge.
- **Multi-hop**: each question needs 2-3 lookups. 32% per hop compounds to about 3%. The reader also runs with `reasoning_effort: none`.

## Spend

| | USD |
|---|---|
| Usable run | 13.27 |
| First run (cache-key bug, superseded) | 11.57 |
| Truncated summaries (not in result file) | 2.31 |
| Retries and partial runs | 0.12 |
| **MemoryAgentBench total** | **27.27** |
| Whole study to date | 28.64 |

## Rules for the remaining arms

1. Commit before every paid run (result files record the git commit).
2. Memory arms: skip `infbench_sum` and `recsys_redial_full`, which cannot be scored or compared:
   `--include_secondary --sources detective_qa,eventqa,factconsolidation_,icl_,longmemeval_s,ruler_qa`
3. Run `--task_prompts` for every arm, or the sensitivity analysis is not fair.
4. Smoke-test Mem0 and Supermemory ingestion with `--num_cases 3` before full runs (LongMemEval is about 52M tokens).
5. Compare arms only on questions every arm answered.
6. Check the errored count; `status: complete` does not mean zero errors.
7. Close the gates in `configs/approvals.yaml` when the study is done.
