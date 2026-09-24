.PHONY: install test test-integration lint typecheck check smoke demo reset-db export-graph migrate download-benchmarks validate-benchmarks eval-suite eval-diagnostic eval-diagnostic-live eval-external eval-report eval-correction validate-external web-install web-build web-dev neo4j-up neo4j-down schema api

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

download-benchmarks:
	uv run python scripts/download_benchmarks.py

validate-benchmarks:
	uv run python -m evals.runners.validate_external --dataset longmemeval --path data/longmemeval/longmemeval_s_cleaned.json
	uv run python -m evals.runners.validate_external --dataset locomo --path data/locomo/locomo10.json
	uv run python -m evals.runners.validate_external --dataset memoryagentbench --path data/memoryagentbench

# Complete research protocol: all 6,157 official questions, live extraction,
# embeddings, answers and official judges, across all three ScopeGraph ablations.
# Set BATCH to a fresh directory for each run; --resume is intentionally omitted.
eval-suite:
	@test -n "$(BATCH)" || (echo 'Set BATCH=results/batches/<run-id>'; exit 1)
	docker compose --profile eval up -d --wait neo4j-eval
	@set -e; for ablation in full no_graph_traversal no_scope_weighting; do \
		NEO4J_URI=bolt://localhost:7688 NEO4J_PASSWORD=scopegraph-eval \
		uv run python -m evals.runners.run_external --dataset longmemeval \
			--path data/longmemeval/longmemeval_s_cleaned.json --live \
			--ablation $$ablation --config configs/experiments.yaml --allow-neo4j-reset \
			--output $(BATCH)/longmemeval-$$ablation.jsonl; \
		NEO4J_URI=bolt://localhost:7688 NEO4J_PASSWORD=scopegraph-eval \
		uv run python -m evals.runners.run_external --dataset locomo \
			--path data/locomo/locomo10.json --live \
			--ablation $$ablation --config configs/experiments.yaml --allow-neo4j-reset \
			--output $(BATCH)/locomo-$$ablation.jsonl; \
		NEO4J_URI=bolt://localhost:7688 NEO4J_PASSWORD=scopegraph-eval \
		uv run python -m evals.runners.run_external --dataset memoryagentbench \
			--path data/memoryagentbench --live \
			--ablation $$ablation --config configs/experiments.yaml --allow-neo4j-reset \
			--output $(BATCH)/memoryagentbench-$$ablation.jsonl; \
	done

web-install:
	npm --prefix web install

web-build:
	npm --prefix web run build

web-dev:
	npm --prefix web run dev

eval-diagnostic:
	docker compose --profile eval up -d --wait neo4j-eval
	NEO4J_URI=bolt://localhost:7688 NEO4J_PASSWORD=scopegraph-eval \
	uv run python -m evals.runners.run_all --dataset cross_scope_mem --config configs/experiments.yaml \
		--allow-neo4j-reset \
		--scenario-count $(if $(SCENARIOS),$(SCENARIOS),1) --difficulty $(if $(DIFFICULTY),$(DIFFICULTY),2) \
		$(if $(LIVE),--live,) $(if $(LIVE_ANSWER),--live-answer,)

# Real end-to-end ScopeGraph run against an isolated Neo4j service. The runner
# clears this evaluation-only database between scenarios so repeated fixture IDs
# cannot leak state across trials.
eval-diagnostic-live:
	docker compose --profile eval up -d --wait neo4j-eval
	NEO4J_URI=bolt://localhost:7688 NEO4J_PASSWORD=scopegraph-eval \
	uv run python -m evals.runners.run_eval --dataset cross_scope_mem \
		--allow-neo4j-reset --config configs/experiments.yaml \
		--scenario-count $(if $(SCENARIOS),$(SCENARIOS),10) \
		--difficulty $(if $(DIFFICULTY),$(DIFFICULTY),3) \
		$(if $(LIVE),--live,) $(if $(LIVE_ANSWER),--live-answer,)

eval-external:
	docker compose --profile eval up -d --wait neo4j-eval
	NEO4J_URI=bolt://localhost:7688 NEO4J_PASSWORD=scopegraph-eval \
	uv run python -m evals.runners.run_external --dataset $(DATASET) --path $(DATA_PATH) \
		--config configs/experiments.yaml --allow-neo4j-reset \
		$(if $(LIVE),--live,) $(if $(LIVE_ANSWER),--live-answer,)

eval-report:
	@test -n "$(BATCH)" || (echo 'Set BATCH=results/batches/<run-id>'; exit 1)
	uv run python -m evals.analysis.run_report $(BATCH)/*.jsonl --output-root $(BATCH)/report

eval-correction:
	docker compose --profile eval up -d --wait neo4j-eval
	NEO4J_URI=bolt://localhost:7688 NEO4J_PASSWORD=scopegraph-eval \
	uv run python -m evals.runners.run_correction_eval --allow-neo4j-reset

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
