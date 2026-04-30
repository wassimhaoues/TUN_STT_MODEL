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
make eval-checkpoint
make analyze-errors
make build-phase05-manifests
make prepare-phase05-b
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

## Phase 03 Full Fine-Tune

Phase 03 reuses the training pipeline for the first full train/valid run, then evaluates the exported best model on both the quick and locked test scopes.

Runbook:

- [docs/phase03_full_finetune_runbook.md](/home/coworky/Study/Deeplearning/TUN_STT_MODEL/docs/phase03_full_finetune_runbook.md)

Main commands:

```bash
python training/train_whisper_small.py --run-type phase03_full_finetune --train-samples 0 --valid-samples 0 --max-steps -1
python training/evaluate_checkpoint.py --model-path outputs/train_runs/<run_name> --samples 20
python training/evaluate_checkpoint.py --model-path outputs/train_runs/<run_name> --samples 0
```

## Phase 04 Evaluation And Error Analysis

Phase 04 adds a repeatable error-analysis script that turns prediction CSVs into bucketed failure reports and a fixed manual-review set.

```bash
python training/analyze_errors.py \
  --predictions-csv reports/runs/<evaluation_run_name>/predictions.csv \
  --source-csv dataset/metadata_test.csv
```

Each analysis run writes next to the evaluation report:

- `reports/runs/<evaluation_run_name>/error_analysis/summary.md`
- `reports/runs/<evaluation_run_name>/error_analysis/bucket_summary.csv`
- `reports/runs/<evaluation_run_name>/error_analysis/detailed_rows.csv`
- `reports/runs/<evaluation_run_name>/error_analysis/manual_review_candidates.csv`

## Phase 05 Targeted Improvements

Phase 05 turns the Phase 4 findings into controlled experiments:

- safer decoding to reduce repetition loops and catastrophic generation failures
- targeted audited train-manifest expansion for code-switched and short clips

Helpful commands:

```bash
python training/evaluate_checkpoint.py --help
python dataset/build_phase05_manifests.py --help
python dataset/prepare_phase05_experiment_b.py --help
```

Runbook:

- [docs/phase05_targeted_improvements_runbook.md](/home/coworky/Study/Deeplearning/TUN_STT_MODEL/docs/phase05_targeted_improvements_runbook.md)
- [docs/phase05_experiment_b_preparation.md](/home/coworky/Study/Deeplearning/TUN_STT_MODEL/docs/phase05_experiment_b_preparation.md)

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
