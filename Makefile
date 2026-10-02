.PHONY: install check lint typecheck test census pilot-select pilot-report verify-data judge-diagnostics

PYTHON ?= .venv/bin/python
export PYTHONPATH := src

install:
	uv pip install --python $(PYTHON) -e ".[dev]"

# Offline and free: lint, strict types, mocked tests. No network, no key.
check: lint typecheck test

lint:
	$(PYTHON) -m ruff check src tests

typecheck:
	$(PYTHON) -m mypy

test:
	$(PYTHON) -m pytest

# Free commands (no model call). Paid commands (run, judge-flip, api-check) are deliberately
# not make targets: invoke `python -m memstudy <command>` after approval, with the gates open in
# configs/approvals.yaml.
census:
	$(PYTHON) -m memstudy census

pilot-select:
	$(PYTHON) -m memstudy pilot-select

verify-data:
	$(PYTHON) -m memstudy verify-data

judge-diagnostics:
	$(PYTHON) -m memstudy judge-diagnostics

pilot-report:
	$(PYTHON) -m memstudy pilot-report
