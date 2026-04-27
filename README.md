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
make train-smoke
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

## Phase 02 Smoke Training

Phase 02 adds a reproducible smoke fine-tuning pipeline for `openai/whisper-small`.
It trains on a deterministic subset, saves checkpoints under `outputs/`, and writes tracked run metadata under `reports/`.

```bash
python training/train_whisper_small.py
python training/train_whisper_small.py --train-samples 256 --valid-samples 64 --max-steps 20
python training/train_whisper_small.py --run-name whisper-small-phase02-gpu --notes "first laptop smoke run"
```

Default smoke behavior:

- train subset: `1000` rows
- valid subset: `200` rows
- precision: `auto` with GPU-aware fallback
- max steps: `60`
- batch config: `4 x grad_accum 4`

Each smoke run writes:

- `outputs/train_runs/<run_name>/`: checkpoints, trainer state, logs
- `reports/runs/<run_name>/summary.md`: tracked Phase 02 report
- `reports/runs/<run_name>/training_config.json`: exact config snapshot
- `reports/runs/<run_name>/environment.json`: hardware and precision snapshot
- `reports/runs/<run_name>/selected_train_manifest.csv`: exact train subset used
- `reports/runs/<run_name>/selected_valid_manifest.csv`: exact valid subset used
- `reports/runs/<run_name>/train_metrics.json`: trainer-side train metrics
- `reports/runs/<run_name>/eval_metrics.json`: validation metrics from generated predictions

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
