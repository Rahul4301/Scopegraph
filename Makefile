.PHONY: install test test-integration lint typecheck check smoke demo reset-db export-graph migrate eval-all eval-neo4j eval-external eval-report eval-correction validate-external web-install web-build web-dev neo4j-up neo4j-down schema api

export PYTHONPATH := src:.
export UV_CACHE_DIR ?= /private/tmp/scopegraph-uv-cache

install:
	uv sync --extra dev

test:
	uv run pytest -m "not integration"

test-integration:
	SCOPEGRAPH_RUN_INTEGRATION=1 uv run pytest -m integration

lint:
	uv run ruff check .

typecheck:
	uv run mypy

check: lint typecheck test

smoke:
	uv run python scripts/smoke_test.py

demo:
	uv run python scripts/run_demo.py

reset-db:
	uv run python scripts/reset_db.py --yes

export-graph:
	uv run python scripts/export_graph.py --output $(if $(OUTPUT),$(OUTPUT),results/graph.json)

migrate:
	uv run python scripts/migrate.py

web-install:
	npm --prefix web install

web-build:
	npm --prefix web run build

web-dev:
	npm --prefix web run dev

eval-all:
	uv run python -m evals.runners.run_all --dataset cross_scope_mem --config configs/experiments.yaml \
		--systems $(if $(SYSTEMS),$(SYSTEMS),vector_memory,flat_graph,two_level_graph,scopegraph) \
		--scenario-count $(if $(SCENARIOS),$(SCENARIOS),1) --difficulty $(if $(DIFFICULTY),$(DIFFICULTY),2) \
		--concurrency $(if $(CONCURRENCY),$(CONCURRENCY),1) \
		$(if $(LIVE),--live,) $(if $(LIVE_ANSWER),--live-answer,)

# Real end-to-end ScopeGraph run against an isolated Neo4j service. The runner
# clears this evaluation-only database between scenarios so repeated fixture IDs
# cannot leak state across trials.
eval-neo4j:
	docker compose --profile eval up -d --wait neo4j-eval
	NEO4J_URI=bolt://localhost:7688 NEO4J_PASSWORD=scopegraph-eval \
	uv run python -m evals.runners.run_eval --dataset cross_scope_mem --system scopegraph \
		--storage neo4j --allow-neo4j-reset --config configs/experiments.yaml \
		--scenario-count $(if $(SCENARIOS),$(SCENARIOS),10) \
		--difficulty $(if $(DIFFICULTY),$(DIFFICULTY),3) \
		$(if $(LIVE),--live,) $(if $(LIVE_ANSWER),--live-answer,)

eval-external:
	uv run python -m evals.runners.run_external --dataset $(DATASET) --path $(DATA_PATH) --system $(SYSTEM) --config configs/experiments.yaml $(if $(LIVE),--live,) $(if $(LIVE_ANSWER),--live-answer,)

eval-report:
	@test -n "$(BATCH)" || (echo 'Set BATCH=results/batches/<run-id>'; exit 1)
	uv run python -m evals.analysis.run_report $(BATCH)/*.jsonl --output-root $(BATCH)/report

eval-correction:
	uv run python -m evals.runners.run_correction_eval

validate-external:
	uv run python -m evals.runners.validate_external --dataset $(DATASET) --path $(DATA_PATH)

neo4j-up:
	docker compose up -d neo4j

neo4j-down:
	docker compose down

schema:
	uv run python scripts/setup_neo4j.py

api:
	uv run uvicorn scopegraph.api.main:app --reload
