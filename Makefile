.PHONY: install test test-integration lint typecheck check web-install web-build web-dev neo4j-up neo4j-down schema api

export PYTHONPATH := src

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

web-install:
	npm --prefix web install

web-build:
	npm --prefix web run build

web-dev:
	npm --prefix web run dev

neo4j-up:
	docker compose up -d neo4j

neo4j-down:
	docker compose down

schema:
	uv run python scripts/setup_neo4j.py

api:
	uv run uvicorn scopegraph.api.main:app --reload
