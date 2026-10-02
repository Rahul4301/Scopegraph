# memstudy

A pre-registered study: when the whole history fits in the model's context window, does a model
still need an external memory layer, and where does one start to pay for itself? The proposal of
record is [PROPOSAL.md](PROPOSAL.md); the registration (hypotheses, falsification conditions,
power curve, cost projection, extensions) is [PREREG.md](PREREG.md).

Status: the harness, the four arms and the pilot tooling are built and tested offline, and the
reader request shape was verified live ($0.0003). No benchmark has been run. `PREREG.md` is a
draft awaiting review and the `prereg-v1` tag. Everything beyond arms A to C on the three
published benchmarks is exploratory (PREREG.md Section 10).

## Arms

| Arm | What it is |
| --- | --- |
| A | No memory: full history first, question last, prompt caching when a history is queried twice or more |
| B | Mem0 open source, pinned `mem0ai==2.2.1`, extraction by GPT-6 Luna (effort none), default embedder, one `user_id` per history |
| C | Plain RAG over verbatim chunks (control) |
| D | Supermemory, self-hosted open-source server (exploratory), same extraction model and embedder as B, one container per history; needs the extra `supermemory` and a running `supermemory-server` |

Same reader (`gpt-6-luna`), template, questions and retrieval token budget (B, C). The sole
grader is `gpt-5-nano`; there is no second grader and no human grading.

## Layout

```
configs/    study.yaml (pre-registered settings), prices.yaml (every price, source, date),
            approvals.yaml (gates, all closed), data_manifest.yaml (checksums)
src/memstudy/
  schema.py budget.py llm.py reader.py judge.py runner.py store.py tokens.py
  execute.py scoring.py chunking.py     (one named run, answer metrics, verbatim chunking)
  arms/      full_context.py mem0_arm.py rag.py supermemory_arm.py
  loaders/   locomo.py longmemeval.py memoryagentbench.py composite.py
  asof.py                               (as-of checkpoints for LoCoMo)
  pilot.py costmodel.py phase0.py preflight.py metering.py datacheck.py cli.py
tests/      offline tests: no network, no key
results/    <system>_<eval>_<YYYYMMDD_HHMMSS>.json (the one file a run writes, never overwritten),
            ledger.jsonl (shared spend log), phase0/, pilot/
```

## Commands

Free (no model call): `census`, `pilot-select`, `verify-data`, `judge-diagnostics`,
`pilot-report`, `run --dry_run`, and `make check` (ruff, strict mypy, offline tests).

### Running a memory system on an eval

```bash
python -m memstudy run --memory_system mem0 --eval locomo --num_cases 1   # smoke test: one conversation (all its questions)
python -m memstudy run --memory_system mem0 --eval locomo                 # full run
python -m memstudy run --memory_system rag --eval memoryagentbench --num_cases 1 --dry_run  # free preview
```

| Flag | Values |
| --- | --- |
| `--memory_system` | `no_memory` (A), `mem0` (B), `rag` (C), `supermemory` (D); the letters work too |
| `--eval` | `locomo`, `longmemeval`, `memoryagentbench`, `composite` (long histories built from LongMemEval-S, 200K to 950K tokens) |
| `--num_cases` | first N cases, in loader order, each with all its primary questions (a case is one LoCoMo conversation, one LongMemEval haystack, one MemoryAgentBench document); omit for the full run |
| `--include_secondary` | also run non-primary cases (LoCoMo adversarial, MemoryAgentBench exploratory) |
| `--workers` | questions answered at the same time (default 4; 1 = one at a time). The first question of each conversation runs alone to warm the prompt cache |
| `--asof` | locomo only: ask each question after the session with its last evidence turn and again at the end; checkpoints share one memory store (exploratory) |
| `--write_granularity` | mem0 only: `turn`, `ten` (the registered default) or `session` messages per ingestion call |
| `--dry_run` | free: list the cases, history tokens and worst-case reader cost; no key, no gate |
| `--key_var` | name of the variable holding the OpenAI key (default `OPENAI_API_KEY`) |

Each question prints as it is answered (question, actual answer, given answer, judge verdict,
token F1, time, running tally), then a summary prints at the end. The result is one file,
`results/<system>_<eval>_<YYYYMMDD_HHMMSS>.json`, for example
`results/mem0_locomo_20261002_093245.json`; it is created exclusively and never overwritten. It holds
`run` (provenance), `summary` (judge accuracy, exact match, substring match, token F1,
gold-in-context, gold-in-store and the stage attribution for memory arms, accuracy by category, total and per-question time, ingestion time, reader and judge
tokens, cache hit rate, context size, cost split into reader, judge, retrieval and ingestion, stored
units, and the case numbers that were wrong), every `cases` entry with its question, gold answer,
given answer, verdict, scores, time, tokens and cost, and `metric_notes`. If a run stops early
(budget cap, API error, Ctrl-C) the file is still written, marked `incomplete`, with every
question answered so far; the run is not resumable.

Paid runs are refused until `prereg-v1` is tagged and the matching gate in
`configs/approvals.yaml` is true: `stage_smoke` for `--num_cases`, `stage_chat_<eval>` for a full
run (`stage_chat_composite` for the composite benchmark; a gate that is absent counts as closed).
Arms B, C and D also need `g1_extra_models` (the embedder besides Luna and nano), and arm D needs
`g2_supermemory_self_hosted`. Smoke runs count against the pilot spend cap. Other paid
commands: `judge-flip`; `api-check --yes` is a two-call, under-$0.001 request check.

```bash
python -m memstudy <command>     # needs PYTHONPATH=src or an editable install
```

Setup: `uv pip install -e ".[dev]"`, then `python -m spacy download en_core_web_sm` (installed).
Keys come from the environment or `.env` and are never printed or logged: `OPENAI_API_KEY`
(`--key_var` or `--key-var` can name a differently named variable). Arm D also reads
`SUPERMEMORY_API_KEY`, the key the local server prints on first boot.

## Guarantees in code

- Result files are created exclusively, so a result is never overwritten.
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
checks use a proxy tokenizer (o200k_base). Mem0's extraction model is GPT-6 Luna, not the library's own default, so Mem0 numbers are not
comparable to Mem0's published ones on that count too. The only chat models are GPT-6 Luna (reader and
extraction) and GPT-5 nano (judge).
