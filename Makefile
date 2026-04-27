PYTHON ?= python3
PIP := $(PYTHON) -m pip
RUFF := $(PYTHON) -m ruff
PYTEST := $(PYTHON) -m pytest

.PHONY: install-runtime install-dev lint format format-check test check validate-dataset check-dataset smoke-whisper baseline clean

install-runtime:
	$(PIP) install -r requirements/runtime.txt

install-dev:
	$(PIP) install -r requirements/dev.txt

lint:
	$(RUFF) check .

format:
	$(RUFF) format .

format-check:
	$(RUFF) format --check .

test:
	$(PYTEST)

check: lint format-check test

validate-dataset:
	$(PYTHON) dataset/validate_dataset.py

check-dataset:
	$(PYTHON) training/check_dataset.py

smoke-whisper:
	$(PYTHON) training/test_whisper_small.py

baseline:
	$(PYTHON) training/baseline_test.py

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__ htmlcov .coverage coverage.xml
