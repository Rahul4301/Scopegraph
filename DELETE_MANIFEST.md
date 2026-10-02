# Delete manifest

Approved by the owner in chat ("get rid of all the scopegraph files that we dont need"). Removed
with `git rm` only. Untracked and ignored files, `data/`, and `.env` are not touched. The original
code stays on branch `main` and in earlier commits of this branch.

| Entry | Reason |
| --- | --- |
| `src/scopegraph/` | ScopeGraph system: scope models, graph repositories, memory, API, backends |
| `evals/` | ScopeGraph eval runners, CrossScopeMem generator, analysis, judges |
| `tests/unit/`, `tests/evals/`, `tests/integration/`, `tests/fixtures/` | Tests for the removed code (Neo4j integration included) |
| `web/` | React Memory Explorer for ScopeGraph |
| `scripts/` | ScopeGraph setup, demo, smoke, and benchmark download scripts |
| `docs/` | ScopeGraph design and protocol docs |
| `configs/experiments.yaml`, `logging.yaml`, `memory.yaml`, `models.example.yaml`, `retrieval.yaml` | ScopeGraph settings (memstudy uses `study.yaml`, `prices.yaml`, `approvals.yaml`) |
| `docker-compose.yml`, `setup.sh` | Neo4j service and ScopeGraph setup |
| `requirements.txt`, `uv.lock` | Dependency pins for the old project; the lock is regenerated for memstudy |
| `CLAUDE.md`, `AGENTS.md`, `RESULTS.md` | ScopeGraph agent notes and claims ledger |
| `results/README.md`, `results/SCHEMA.md`, and the third tracked file under `results/` | ScopeGraph run table and schema |

Kept: `LICENSE`, `Makefile`, `.env.example`, `.gitignore`, `pyproject.toml`, `README.md`
(all rewritten or generic), and everything new under `src/memstudy/`, `tests/memstudy/`, `configs/`.
The kept `Makefile` and `.env.example` still describe ScopeGraph targets and Neo4j variables and
need a separate cleanup.
