# ScopeGraph — working notes for humans and AI agents

`CLAUDE.md` and `AGENTS.md` are identical; edit both (a test enforces it).

ScopeGraph is a research system testing whether explicit session / project / global memory scopes
reduce cross-context retrieval errors in long-running LLM agents, and whether editing memory
directly corrects errors more durably than a conversational correction. Start with
[docs/README.md](docs/README.md) and [RESULTS.md](RESULTS.md).

## Repo map

| Path | What is there |
| --- | --- |
| `src/scopegraph/` | The system: `models/` (pydantic), `graph/` (Neo4j + in-memory repositories), `memory/` (write path, retriever, corrections), `llm/` (extraction, answering, transport), `embeddings/`, `api/` (FastAPI), `backends/`, `integrations/`, `observability/` |
| `evals/` | `adapters/` (datasets), `scenarios/` (CrossScopeMem generator), `runners/` (`run_eval`, `run_all`, `run_external`, `run_correction_eval`), `analysis/` (scoring, CIs, reports, tables, results index and claims check), `judges/`, `metrics/` |
| `tests/` | `unit/`, `evals/`, `integration/` (needs Neo4j, skipped by default) |
| `configs/` | `experiments.yaml` (eval settings), `retrieval.yaml`, `memory.yaml`, `models.example.yaml` |
| `scripts/` | Setup, demo, smoke test, benchmark download |
| `web/` | React Memory Explorer |
| `docs/` | Design and protocol docs (index: `docs/README.md`) |
| `results/` | Run outputs (mostly git-ignored): `results/README.md` (run table), `results/SCHEMA.md` |
| `RESULTS.md` | Claims ledger: every reported number, source file, command, caveat |
| `data/` | Downloaded benchmark data (git-ignored) |

## Make targets

| Target | Does |
| --- | --- |
| `make check` | `ruff` + `mypy` (strict, `src/scopegraph`) + `pytest -m "not integration"`. Offline, free. |
| `make test` / `lint` / `typecheck` / `test-integration` | Parts of `check`; integration needs `SCOPEGRAPH_RUN_INTEGRATION=1` and Neo4j. |
| `make smoke` | Credential-free end-to-end smoke test. |
| `make eval-ablation` | CrossScopeMem batch with all seven conditions (default 10 accounts, difficulty 3). `STORAGE=memory` for an offline run; `LIVE=1` / `LIVE_ANSWER=1` make paid calls. |
| `make eval-report BATCH=<dir>` | Processed JSON, tables, CIs, McNemar, latency, LoCoMo diagnostics. Refuses to overwrite a report. |
| `make eval-smoke` | Live LoCoMo run (`CASES=N`, `ABLATION=vector_only`, `OUTPUT`, `EXTRACTION_CACHE`). **Paid.** |
| `make eval-diagnostic` / `eval-diagnostic-live` / `eval-external` / `eval-suite` / `eval-correction` | Older entry points; live variants are **paid**. |
| `make results-index` | Print the derived run table; fails if a run is missing from `results/README.md`. |
| `make claims-check` | Verify `RESULTS.md` rows, cited files, numbers, and wording. |
| `make neo4j-up` / `neo4j-down` / `migrate` / `schema` / `api` / `web-dev` | Local services. |

## Rules

1. **Never modify existing files under `results/`.** Add new runs and reports beside them; rename
   nothing. Reports and runs refuse to overwrite; do not bypass that without being told to.
2. **No pushes, deletions, or paid/live runs without the user's explicit approval.** A paid run is
   anything that calls the LLM, embedding, or judge APIs (`LIVE=1`, `LIVE_ANSWER=1`, `--live*`,
   `eval-smoke`, `eval-suite`, `eval-external`, `eval-diagnostic-live`). State the plan, cost and
   time estimate first, then wait.
3. **Never print `.env` values** (keys, URLs, passwords). Do not paste them into logs, docs, or
   commits.
4. **No superiority or production-readiness claims.** Say what was measured, on what, with what
   caveat. `make claims-check` rejects phrases such as "outperforms" and "production-ready".
5. Keep **scope-isolation** evidence (CrossScopeMem only) separate from **retrieval-quality**
   evidence (external benchmarks), and never mix their latencies.
6. Fix the existing code path rather than adding parallel scripts or tools; do not add features
   that multiply paid API calls unless asked.
7. Differences are reported **full minus control**; intervals resample whole accounts; fewer than
   10 accounts is underpowered.

## Definition of done

- `make check` passes (ruff, mypy, pytest) and the change has a test that fails without it.
- New or changed behaviour is covered in the relevant doc (`docs/`, `evals/README.md`,
  `results/SCHEMA.md`) and the docs index if a doc was added.
- Any new number is in `RESULTS.md` with source file, command and caveat; `make claims-check` and
  `make results-index` pass.
- New runs have a `run.json`; no existing result file was edited.
- The final message lists files changed, tests added, approvals still needed, and which ledger
  claims remain unsupported.
