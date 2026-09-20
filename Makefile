.PHONY: install test test-integration lint typecheck check smoke demo reset-db export-graph migrate eval-all eval-report eval-correction validate-external web-install web-build web-dev neo4j-up neo4j-down schema api

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
	uv run python -m evals.runners.run_all --dataset cross_scope_mem --config configs/experiments.yaml

eval-report:
	uv run python -m evals.analysis.run_report results/raw/*.jsonl

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
