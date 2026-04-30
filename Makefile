PYTHON ?= python3
PIP := $(PYTHON) -m pip
RUFF := $(PYTHON) -m ruff
PYTEST := $(PYTHON) -m pytest

.PHONY: install-runtime install-dev lint format format-check test check validate-dataset check-dataset smoke-whisper baseline train-smoke eval-checkpoint analyze-errors build-phase05-manifests prepare-phase05-b transcribe-one clean

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

train-smoke:
	$(PYTHON) training/train_whisper_small.py

eval-checkpoint:
	$(PYTHON) training/evaluate_checkpoint.py --help

analyze-errors:
	$(PYTHON) training/analyze_errors.py --help

build-phase05-manifests:
	$(PYTHON) dataset/build_phase05_manifests.py --help

prepare-phase05-b:
	$(PYTHON) dataset/prepare_phase05_experiment_b.py --help

transcribe-one:
	$(PYTHON) training/transcribe_one_audio.py --help

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__ htmlcov .coverage coverage.xml
