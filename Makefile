.PHONY: install eval-smoke eval-ablation results-index claims-check test test-integration lint typecheck check smoke demo reset-db export-graph migrate download-benchmarks validate-benchmarks eval-suite eval-diagnostic eval-diagnostic-live eval-external eval-report eval-correction validate-external web-install web-build web-dev neo4j-up neo4j-down schema api

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

# Complete external-validity protocol: all 6,157 official questions through full
# ScopeGraph with live extraction, embeddings, answers, and official judges.
# Architecture controls and component ablations run only in CrossScopeMem.
# Set BATCH to a fresh directory for each run; --resume is intentionally omitted.
eval-suite:
	@test -n "$(BATCH)" || (echo 'Set BATCH=results/batches/<run-id>'; exit 1)
	docker compose --profile eval up -d --wait neo4j-eval
	NEO4J_URI=bolt://localhost:7688 NEO4J_PASSWORD=scopegraph-eval \
		uv run python -m evals.runners.run_external --dataset longmemeval \
			--path data/longmemeval/longmemeval_s_cleaned.json --live \
			--extraction-cache $(BATCH)/longmemeval-extractions.json \
			--ablation full --config configs/experiments.yaml --allow-neo4j-reset \
			--output $(BATCH)/longmemeval-full.jsonl
	NEO4J_URI=bolt://localhost:7688 NEO4J_PASSWORD=scopegraph-eval \
		uv run python -m evals.runners.run_external --dataset locomo \
			--path data/locomo/locomo10.json --live \
			--extraction-cache $(BATCH)/locomo-extractions.json \
			--ablation full --config configs/experiments.yaml --allow-neo4j-reset \
			--output $(BATCH)/locomo-full.jsonl
	NEO4J_URI=bolt://localhost:7688 NEO4J_PASSWORD=scopegraph-eval \
		uv run python -m evals.runners.run_external --dataset memoryagentbench \
			--path data/memoryagentbench --live \
			--extraction-cache $(BATCH)/memoryagentbench-extractions.json \
			--ablation full --config configs/experiments.yaml --allow-neo4j-reset \
			--output $(BATCH)/memoryagentbench-full.jsonl

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
	uv run python -m evals.runners.run_all --dataset cross_scope_mem \
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

# Live LoCoMo run on the first CASES conversations (default 1 = 199 questions; 10 = all).
# OUTPUT defaults to results/smoke/mm_dd__hh_mm.jsonl (local time; commit and dataset are in
# the .run.json beside it) and an existing file is never overwritten. A readable
# <name>.report.md and <name>.questions.csv are written beside it when the run finishes.
# ABLATION=vector_only runs the LoCoMo-only plain-vector baseline (protocol amendment,
# docs/benchmark_protocol.md). EXTRACTION_CACHE defaults to a fresh per-run file so a run
# never edits an existing cache; pass one explicitly to share extraction between the full
# and vector_only runs. Resume an interrupted run with OUTPUT=<same file> RESUME=1.
eval-smoke:
	docker compose --profile eval up -d --wait neo4j-eval
	@out="$(if $(OUTPUT),$(OUTPUT),results/smoke/$$(date +%m_%d__%H_%M).jsonl)"; \
	cache="$(if $(EXTRACTION_CACHE),$(EXTRACTION_CACHE),$${out%.jsonl}-extractions.json)"; \
	echo "output: $$out"; echo "extraction cache: $$cache"; \
	NEO4J_URI=bolt://localhost:7688 NEO4J_PASSWORD=scopegraph-eval \
	uv run python -m evals.runners.run_external --dataset locomo --path data/locomo/locomo10.json \
		--case-limit $(if $(CASES),$(CASES),1) $(if $(LIMIT),--limit $(LIMIT),) --live \
		--config configs/experiments.yaml --allow-neo4j-reset \
		--extraction-cache "$$cache" --output "$$out" \
		$(if $(ABLATION),--ablation $(ABLATION),) $(if $(RESUME),--resume,)

# Pilot-grade CrossScopeMem batch: all seven conditions (full, vector_only_control,
# vector_scope_filter, flat_graph_control, two_level_control, no_graph_traversal,
# no_temporal_status) over SCENARIOS accounts (default 10, the pilot minimum; the proposal
# targets 40-60). Writes a NEW results/batches/<timestamp>/ with run.json; then run
# `make eval-report BATCH=<that directory>`. STORAGE=memory is an offline smoke (no Neo4j).
STORAGE ?= neo4j
eval-ablation:
	@test "$(if $(SCENARIOS),$(SCENARIOS),10)" -ge 10 || test -n "$(SMALL)" || \
		(echo 'Pilot needs SCENARIOS>=10 (set SMALL=1 for a labelled smoke run)'; exit 1)
	@if [ "$(STORAGE)" = "neo4j" ]; then docker compose --profile eval up -d --wait neo4j-eval; fi
	NEO4J_URI=bolt://localhost:7688 NEO4J_PASSWORD=scopegraph-eval \
	uv run python -m evals.runners.run_all --dataset cross_scope_mem --config configs/experiments.yaml \
		--storage $(STORAGE) --allow-neo4j-reset --output results/batches \
		--scenario-count $(if $(SCENARIOS),$(SCENARIOS),10) \
		--difficulty $(if $(DIFFICULTY),$(DIFFICULTY),3) \
		$(if $(LIVE),--live,) $(if $(LIVE_ANSWER),--live-answer,)

# Derived run table for results/ (never edits results); fails if a run is missing from the
# curated table in results/README.md.
results-index:
	uv run python -m evals.analysis.tables index results

# Verify RESULTS.md: complete rows, existing source files, machine-checked numbers match,
# and no superiority or production-readiness wording in the docs.
claims-check:
	uv run python -m evals.analysis.tables claims RESULTS.md

eval-report:
	@test -n "$(BATCH)" || (echo 'Set BATCH=results/batches/<run-id>'; exit 1)
	uv run python -m evals.analysis.run_report $(BATCH)/*.jsonl --output-root $(BATCH)/report \
		$(if $(LOCOMO_DATA),--locomo-data $(LOCOMO_DATA),)

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
