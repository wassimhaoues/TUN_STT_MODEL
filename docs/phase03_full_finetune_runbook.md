# Phase 03 Full Fine-Tune Runbook

This runbook executes the first full fine-tuning phase end to end:

1. full training on `metadata_train.csv` and `metadata_valid.csv`
2. quick evaluation on `test_head_20`
3. locked evaluation on the full `metadata_test.csv`

## Assumptions

- the Python environment is `train-env`
- Phase 01 normalization is already in place
- Phase 02 smoke training already passed on this machine

## Commands

```bash
cd /home/coworky/Study/Deeplearning/TUN_STT_MODEL
source train-env/bin/activate
python -m pip install -r requirements/runtime.txt -r requirements/dev.txt
make lint
make test
```

Choose a run name once and reuse it:

```bash
export RUN_NAME="whisper-small-phase03-full-$(date +%Y%m%d-%H%M%S)"
```

Run the full fine-tune:

```bash
python training/train_whisper_small.py \
  --run-name "$RUN_NAME" \
  --run-type phase03_full_finetune \
  --notes "Phase 03 first full fine-tune on full normalized train/valid split" \
  --train-samples 0 \
  --valid-samples 0 \
  --max-steps -1 \
  --num-train-epochs 3 \
  --per-device-train-batch-size 4 \
  --per-device-eval-batch-size 4 \
  --gradient-accumulation-steps 4 \
  --learning-rate 1e-5 \
  --warmup-ratio 0.1 \
  --eval-steps 250 \
  --save-steps 250 \
  --logging-steps 25 \
  --save-total-limit 3 \
  --dataloader-num-workers 2 \
  --gradient-checkpointing \
  --group-by-length \
  --precision auto
```

Important note:

- the exported best model is saved to `outputs/train_runs/$RUN_NAME`
- you do not need to point evaluation at a checkpoint subfolder unless you explicitly want that

Quick comparison run on `test_head_20`:

```bash
python training/evaluate_checkpoint.py \
  --model-path "outputs/train_runs/$RUN_NAME" \
  --run-name "${RUN_NAME}-quick-eval" \
  --run-type phase03_quick_eval \
  --source-csv dataset/metadata_test.csv \
  --samples 20 \
  --notes "Phase 03 quick benchmark on test_head_20"
```

Locked full test evaluation:

```bash
python training/evaluate_checkpoint.py \
  --model-path "outputs/train_runs/$RUN_NAME" \
  --run-name "${RUN_NAME}-locked-test" \
  --run-type phase03_full_eval \
  --source-csv dataset/metadata_test.csv \
  --samples 0 \
  --notes "Phase 03 locked full test evaluation"
```

Optional quick inspection of the recorded outputs:

```bash
sed -n '1,220p' "reports/runs/$RUN_NAME/summary.md"
sed -n '1,220p' "reports/runs/${RUN_NAME}-quick-eval/summary.md"
sed -n '1,220p' "reports/runs/${RUN_NAME}-locked-test/summary.md"
tail -n 10 reports/experiment_history.csv
```

## Expected Artifacts

Training run:

- `outputs/train_runs/$RUN_NAME/`
- `reports/runs/$RUN_NAME/summary.md`
- `reports/runs/$RUN_NAME/training_config.json`
- `reports/runs/$RUN_NAME/environment.json`
- `reports/runs/$RUN_NAME/train_metrics.json`
- `reports/runs/$RUN_NAME/eval_metrics.json`

Evaluation runs:

- `reports/runs/${RUN_NAME}-quick-eval/summary.md`
- `reports/runs/${RUN_NAME}-quick-eval/predictions.csv`
- `reports/runs/${RUN_NAME}-locked-test/summary.md`
- `reports/runs/${RUN_NAME}-locked-test/predictions.csv`

## Phase 03 Completion Checklist

- full training finished without OOM
- best model exported under `outputs/train_runs/$RUN_NAME`
- quick benchmark completed on `test_head_20`
- locked evaluation completed on full `metadata_test.csv`
- all three runs are appended to `reports/experiment_history.csv`
