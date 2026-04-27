# TUN STT Model

Development workspace for preparing and fine-tuning Whisper Small to better transcribe Tunisian Derja.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/runtime.txt -r requirements/dev.txt
pre-commit install
```

## Common Commands

```bash
make lint
make format
make test
make validate-dataset
make check-dataset
make smoke-whisper
make baseline
```

## Baseline Tracking

The baseline script now records a tracked experiment report in `reports/` every time it runs.

```bash
python training/baseline_test.py
python training/baseline_test.py --run-name baseline-start --notes "raw whisper-small before fine-tuning"
```

Each run writes:

- `reports/experiment_history.csv`: one row per experiment run
- `reports/runs/<run_name>/summary.md`: readable baseline summary
- `reports/runs/<run_name>/predictions.csv`: per-sample predictions for that run

## Project Layout

- `dataset/`: metadata preparation, validation, and split scripts.
- `reports/`: tracked baseline and experiment history for later comparison.
- `training/`: model smoke tests and baseline transcription checks.
- `tests/`: fast repo sanity checks for CI and local development.
- `.vscode/`: editor settings, recommended extensions, and one-click tasks.

## Notes

- `requirements/runtime.txt` contains the Python libraries used by the dataset and training scripts.
- `requirements/dev.txt` contains developer tooling only.
- For GPU-specific PyTorch installs, use the official PyTorch command if you need a different CUDA build than the default wheel.
