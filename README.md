# memstudy

A pre-registered study: when the whole history fits in the model's context window, does a model
still need an external memory layer? Plan, hypotheses, falsification conditions and the cost
projection: [PREREG.md](PREREG.md).

Status: the harness, the three arms and the pilot tooling are built and tested offline, and the
reader request shape was verified live ($0.0003). No benchmark has been run. `PREREG.md` is a draft awaiting review and the `prereg-v1` tag.

## Arms

| Arm | What it is |
| --- | --- |
| A | No memory: full history first, question last, prompt caching when a history is queried twice or more |
| B | Mem0 open source, pinned `mem0ai==2.2.1`, default models, one `user_id` per history |
| C | Plain RAG over verbatim chunks (control) |

Same reader (`gpt-6-luna`), template, questions and retrieval token budget (B, C). The sole
grader is `gpt-5-nano`; there is no second grader and no human grading.

## Layout

```
configs/    study.yaml (pre-registered settings), prices.yaml (every price, source, date),
            approvals.yaml (gates, all closed), data_manifest.yaml (checksums)
src/memstudy/
  schema.py budget.py llm.py reader.py judge.py runner.py store.py tokens.py
  execute.py scoring.py chunking.py     (one named run, answer metrics, verbatim chunking)
  arms/      full_context.py mem0_arm.py rag.py
  loaders/   locomo.py longmemeval.py memoryagentbench.py
  pilot.py costmodel.py phase0.py preflight.py metering.py datacheck.py cli.py
tests/      offline tests: no network, no key
results/    <system>_<eval>[_<n>].json (one readable file per run), runs/<name>/ (raw records,
            never overwritten), ledger.jsonl, phase0/, pilot/
```

## Commands

Free (no model call): `census`, `pilot-select`, `verify-data`, `judge-diagnostics`,
`pilot-report`, `run --dry_run`, and `make check` (ruff, strict mypy, offline tests).

### Running a memory system on an eval

```bash
python -m memstudy run --memory_system mem0 --eval locomo --num_cases 1   # smoke test, one case
python -m memstudy run --memory_system mem0 --eval locomo                 # full run
python -m memstudy run --memory_system rag --eval memoryagentbench --num_cases 1 --dry_run  # free preview
```

| Flag | Values |
| --- | --- |
| `--memory_system` | `no_memory` (A), `mem0` (B), `rag` (C); the letters work too |
| `--eval` | `locomo`, `longmemeval`, `memoryagentbench` |
| `--num_cases` | first N primary cases, in loader order; omit for the full run |
| `--include_secondary` | also run non-primary cases (LoCoMo adversarial, MemoryAgentBench exploratory) |
| `--dry_run` | free: list the cases, history tokens and worst-case reader cost; no key, no gate |
| `--key_var` | name of the variable holding the OpenAI key (default `OPENAI_API_KEY`) |

The result is `results/<system>_<eval>_<N>.json`, or `results/<system>_<eval>.json` for a full
run, for example `results/mem0_locomo_1.json`. It is created exclusively and never overwritten:
running the same command again refuses until you move the file. It holds `run` (provenance), `summary`
(judge accuracy, exact match, substring match, token F1, gold-in-context, accuracy by category,
reader and judge tokens, cache hit rate, context size, latency p50/p95 in ms, cost split into
reader, judge, retrieval and ingestion, ingestion time and stored units), every `cases` entry with
its answer and per-case metrics, and `metric_notes`. If a run stops early (budget cap, API error)
the raw records stay under `results/runs/<name>/` and running the same command resumes from them.

Paid runs are refused until `prereg-v1` is tagged and the matching gate in
`configs/approvals.yaml` is true: `stage_smoke` for `--num_cases`, `stage_chat_<eval>` for a full
run. Arms B and C also need `g1_extra_models` (extraction model and embedder besides Luna and
nano). Smoke runs count against the pilot spend cap. Other paid
commands: `judge-flip`; `api-check --yes` is a two-call, under-$0.001 request check.

```bash
python -m memstudy <command>     # needs PYTHONPATH=src or an editable install
```

Setup: `uv pip install -e ".[dev]"`, then `python -m spacy download en_core_web_sm` (installed).
Keys come from the environment or `.env` and are never printed or logged: `OPENAI_API_KEY`
(`--key_var` or `--key-var` can name a differently named variable).

## Guarantees in code

- Raw results are created exclusively, so a result is never overwritten; runs resume by skipping
  items that already have a raw file.
- No truncation: a history that does not fit is recorded as `does_not_fit`. Retrieved context is
  filled with whole items under one token budget; nothing is cut mid-item.
- The cached-token count returned by the API is logged on every reader call.
- The budget guard refuses any call that could exceed a cap.
- Mem0 isolation is tested (`tests/test_mem0_isolation.py`).

## Limitations

Single seed. Single reader model and one vendor for reader and judge. One LLM grader whose error
and family-level bias are not independently measured. Only ten LoCoMo histories, so history-level
intervals are wide. Conversational benchmarks only. Mem0 OSS rather than the hosted Platform, so
numbers are not comparable to Mem0's published ones. The reader is pinned by alias only; the served snapshot is logged. Token fit
checks use a proxy tokenizer (o200k_base). Mem0's default extraction model `gpt-5-mini` is marked
Deprecated.
