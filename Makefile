.PHONY: install check lint typecheck test census pilot-select judge-export judge-eval coding-baseline pilot-report

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

# Free commands (no model call). Paid commands (run, judge-rerun, coding-run) are deliberately
# not make targets: invoke `python -m memstudy <command>` after approval, with the gates open in
# configs/approvals.yaml.
census:
	$(PYTHON) -m memstudy census

pilot-select:
	$(PYTHON) -m memstudy pilot-select

judge-export:
	$(PYTHON) -m memstudy judge-export

judge-eval:
	@test -n "$(LABELED)" || (echo 'Set LABELED=<path to hand-labeled csv>'; exit 1)
	$(PYTHON) -m memstudy judge-eval $(LABELED)

coding-baseline:
	$(PYTHON) -m memstudy coding-baseline

pilot-report:
	$(PYTHON) -m memstudy pilot-report
