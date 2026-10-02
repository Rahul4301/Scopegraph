# memstudy

A pre-registered study of one question: does a model need an external memory layer when the
whole history fits in context? Plan, falsification condition, and cost caps:
[PREREG.md](PREREG.md).

Status: Phase 1 (harness) and the Phase 2 pilot tooling are built and tested offline. No paid
call has been made. Phase 0 documents are drafts awaiting the owner's review and the
`prereg-v1` tag.

## Arms

| Arm | What it is |
| --- | --- |
| A | Full history first, question last, prompt caching when a history is queried twice or more |
| B | Mem0 open source, pinned `mem0ai==2.2.1`, one `user_id` per history |
| C | Plain RAG over verbatim chunks (control) |

All arms use the same reader (`gpt-6-luna`) and prompt template. LoCoMo and LongMemEval-S are
judged by `gpt-5-nano`; the coding benchmark is graded by its own tests.

## Layout

```
configs/    study.yaml (pre-registered settings), prices.yaml (every price, source, date),
            approvals.yaml (gates, all closed)
src/memstudy/
  schema.py budget.py llm.py reader.py judge.py runner.py store.py
  arms/      full_context.py mem0_arm.py rag.py
  loaders/   locomo.py longmemeval.py swectx.py
  pilot.py coding.py costmodel.py phase0.py preflight.py metering.py cli.py
tests/memstudy/   offline tests, no network, no key
results/    ledger.jsonl, phase0/, pilot/, full/  (raw/<arm>/<bench>/<item>.json never overwritten)
```

## Commands

Free (no model call):

```bash
python -m memstudy census          # token census of every history
python -m memstudy pilot-select    # deterministic pilot sample
python -m memstudy judge-export    # 100-answer hand-check, verdicts hidden
python -m memstudy judge-eval labeled.csv
python -m memstudy coding-baseline
python -m memstudy pilot-report
pytest                             # 85 offline tests
```

Paid (refused until `prereg-v1` is tagged and the matching gate in `configs/approvals.yaml` is
true): `run`, `judge-rerun`, `coding-run`. Arms B and C also need gate `g1_extra_models`.

Setup: `uv pip install -e ".[dev]"`, then `python -m spacy download en_core_web_sm` (Mem0 needs
it for hybrid retrieval; arm B refuses to start without it). `OPENAI_API_KEY` is read from the
environment or `.env` and is never printed or logged.

## Guarantees in code

- Raw results are created exclusively (`open(..., "x")`), so a result is never overwritten; runs
  resume by skipping items that already have a raw file.
- No truncation: a history that does not fit is recorded as `does_not_fit`. The only caps are
  explicit and marked: coding tool-output cap, and the coding memory query cap (flagged in the
  record).
- The cached-token count returned by the API is logged on every reader call.
- The budget guard refuses any call that could exceed a cap.
- Mem0 isolation is tested against the real Mem0 class (`tests/memstudy/test_mem0_isolation.py`).

## Limitations

Single seed. Single reader model and one vendor (Supermemory, arm D, is not run). The judge
differs from each benchmark's default judge. Mem0 OSS rather than the hosted Platform, so numbers
are not comparable to Mem0's published ones. The reader is pinned by alias only; the served
snapshot is logged. Token fit checks use a proxy tokenizer (o200k_base). The coding benchmark is
SWE Context Bench (VibeMemBench has no released code), and its history is gold-patch experience,
not trajectories. The Docker sandbox and SWE-bench grader are written but have not run.
LoCoMo has 10 histories, so its history-level confidence intervals are wide.
