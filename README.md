# memstudy

A pre-registered study: when the whole history fits in the model's context window, does a model
still need an external memory layer? Plan, hypotheses, falsification conditions and the cost
projection: [PREREG.md](PREREG.md).

Status: the harness, all four arms and the pilot tooling are built and tested offline, and the
reader request shape was verified live ($0.0003). No benchmark has been run. Arm C (Supermemory)
has not touched the real service. `PREREG.md` is a draft awaiting review and the `prereg-v1` tag.

## Arms

| Arm | What it is |
| --- | --- |
| A | No memory: full history first, question last, prompt caching when a history is queried twice or more |
| B | Mem0 open source, pinned `mem0ai==2.2.1`, default models, one `user_id` per history |
| C | Supermemory hosted API, `supermemory==3.62.0`, one container per history |
| D | Plain RAG over verbatim chunks (control) |

Same reader (`gpt-6-luna`), template, questions and retrieval token budget (B, C, D). The sole
grader is `gpt-5-nano`; there is no second grader and no human grading.

## Layout

```
configs/    study.yaml (pre-registered settings), prices.yaml (every price, source, date),
            approvals.yaml (gates, all closed), data_manifest.yaml (checksums)
src/memstudy/
  schema.py budget.py llm.py reader.py judge.py runner.py store.py tokens.py
  arms/      full_context.py mem0_arm.py supermemory_arm.py rag.py
  loaders/   locomo.py longmemeval.py
  pilot.py costmodel.py phase0.py preflight.py metering.py datacheck.py cli.py
tests/      offline tests: no network, no key
results/    ledger.jsonl, phase0/, pilot/, full/  (raw/<arm>/<bench>/<item>.json never overwritten)
```

## Commands

Free (no model call): `census`, `pilot-select`, `verify-data`, `judge-diagnostics`,
`pilot-report`, and `make check` (ruff, strict mypy, 93 offline tests).

Paid, refused until `prereg-v1` is tagged and the gate in `configs/approvals.yaml` is true:
`run` (arms A to D), `judge-flip`. `api-check --yes` is a two-call, under-$0.001 request check.
Arm B and D also need gate `g1_extra_models` (extraction model and embedder besides Luna and nano);
arm C needs `g2_supermemory`.

```bash
python -m memstudy <command>     # needs PYTHONPATH=src or an editable install
```

Setup: `uv pip install -e ".[dev]"`, then `python -m spacy download en_core_web_sm` (installed).
Keys come from the environment or `.env` and are never printed or logged: `OPENAI_API_KEY`
(`--key-var` can name a differently named variable) and `SUPERMEMORY_API_KEY`.

## Guarantees in code

- Raw results are created exclusively, so a result is never overwritten; runs resume by skipping
  items that already have a raw file.
- No truncation: a history that does not fit is recorded as `does_not_fit`. Retrieved context is
  filled with whole items under one token budget; nothing is cut mid-item.
- The cached-token count returned by the API is logged on every reader call.
- The budget guard refuses any call that could exceed a cap; Supermemory spend is an estimate
  from its published rate card.
- Mem0 and Supermemory isolation are tested (`tests/test_mem0_isolation.py`,
  `tests/test_supermemory.py`).

## Limitations

Single seed. Single reader model and one vendor for reader and judge. One LLM grader whose error
and family-level bias are not independently measured. Only ten LoCoMo histories, so history-level
intervals are wide. Conversational benchmarks only. Mem0 OSS rather than the hosted Platform, so
numbers are not comparable to Mem0's published ones. Supermemory's internal models and exact costs
are not observable. The reader is pinned by alias only; the served snapshot is logged. Token fit
checks use a proxy tokenizer (o200k_base). Mem0's default extraction model `gpt-5-mini` is marked
Deprecated.
