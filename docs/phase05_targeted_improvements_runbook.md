# Phase 05 Targeted Improvements Runbook

This phase is split into controlled experiment families so we do not mix too many variables at once.

## What Changed In The Repo

### 1. Safer decoding for repetition failures

`training/train_whisper_small.py` and `training/evaluate_checkpoint.py` now support tracked decoding presets and explicit generation controls:

- `--decoding-preset`
- `--generation-max-length`
- `--generation-num-beams`
- `--generation-length-penalty`
- `--generation-no-repeat-ngram-size`
- `--generation-repetition-penalty`

The new preset `phase05_safe_decode_v1` is designed to target the Phase 4 looping errors:

- `num_beams=3`
- `no_repeat_ngram_size=3`
- `repetition_penalty=1.10`
- `length_penalty=1.0`

### 2. Targeted manifest expansion for code-switching and short clips

`dataset/build_phase05_manifests.py` creates a new training manifest that duplicates:

- code-switched rows
- short clips

The validation set stays unchanged so the comparison remains honest.

## Recommended Experiment Order

Run these in order. Each one answers a different question.

### Experiment A: Decode-only test on the existing Phase 03 model

Question:

- Can safer decoding reduce catastrophic loops without retraining?

Commands:

```bash
export PHASE3_RUN="whisper-small-phase03-full-20260428-001840"

python training/evaluate_checkpoint.py \
  --model-path "outputs/train_runs/$PHASE3_RUN" \
  --run-name "${PHASE3_RUN}-phase05-safe-decode-quick" \
  --run-type phase05_decode_only \
  --source-csv dataset/metadata_test.csv \
  --samples 20 \
  --decoding-preset phase05_safe_decode_v1 \
  --notes "Phase 05 decode-only quick evaluation with safer decoding"

python training/evaluate_checkpoint.py \
  --model-path "outputs/train_runs/$PHASE3_RUN" \
  --run-name "${PHASE3_RUN}-phase05-safe-decode-locked" \
  --run-type phase05_decode_only \
  --source-csv dataset/metadata_test.csv \
  --samples 0 \
  --decoding-preset phase05_safe_decode_v1 \
  --notes "Phase 05 decode-only locked evaluation with safer decoding"

python training/analyze_errors.py \
  --predictions-csv "reports/runs/${PHASE3_RUN}-phase05-safe-decode-locked/predictions.csv" \
  --source-csv dataset/metadata_test.csv
```

Success signal:

- fewer `repeated_token_hallucination`
- fewer `catastrophic_looping`
- same or better locked-test WER/CER

### Experiment B: Audited data emphasis on code-switched and short clips

Question:

- Does emphasizing code-switched and short training examples improve the hardest regimes from Phase 4?

Before building the manifest:

- regenerate the Phase 05 prep pack after any dataset repair
- review `dataset/audits/phase05_experiment_b_training_audit.csv`
- only mark `keep_for_phase05_boost=yes` for rows you trust
- use `transcript_action=exclude` or `audio_action=exclude` to drop bad rows from the Experiment B train manifest
- use `transcript_action=fix_text` plus `corrected_text` when you have a trusted correction

Commands:

```bash
python dataset/build_phase05_manifests.py \
  --experiment-name phase05-codeswitch-short-boost-audited-v1 \
  --audit-csv dataset/audits/phase05_experiment_b_training_audit.csv \
  --code-switch-boost-factor 2 \
  --short-clip-boost-factor 2 \
  --short-clip-threshold 3.0
```

This writes:

- `dataset/phase05_manifests/phase05-codeswitch-short-boost-audited-v1/metadata_train.csv`
- `dataset/phase05_manifests/phase05-codeswitch-short-boost-audited-v1/metadata_valid.csv`
- `reports/phase05_data_strategies/phase05-codeswitch-short-boost-audited-v1/summary.md`

Train with the new manifests while keeping the Phase 03 hyperparameters stable:

```bash
export RUN_NAME="whisper-small-phase05-data-boost-$(date +%Y%m%d-%H%M%S)"

python training/train_whisper_small.py \
  --run-name "$RUN_NAME" \
  --run-type phase05_data_boost \
  --train-csv dataset/phase05_manifests/phase05-codeswitch-short-boost-audited-v1/metadata_train.csv \
  --valid-csv dataset/phase05_manifests/phase05-codeswitch-short-boost-audited-v1/metadata_valid.csv \
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
  --precision auto \
  --decoding-preset standard \
  --notes "Phase 05 data-only run with audited code-switched and short-clip emphasis"
```

Then evaluate it with the same decoding policy as Phase 03 first:

```bash
python training/evaluate_checkpoint.py \
  --model-path "outputs/train_runs/$RUN_NAME" \
  --run-name "${RUN_NAME}-quick-eval" \
  --run-type phase05_data_boost_eval \
  --source-csv dataset/metadata_test.csv \
  --samples 20 \
  --decoding-preset standard \
  --notes "Phase 05 data-only quick evaluation"

python training/evaluate_checkpoint.py \
  --model-path "outputs/train_runs/$RUN_NAME" \
  --run-name "${RUN_NAME}-locked-test" \
  --run-type phase05_data_boost_eval \
  --source-csv dataset/metadata_test.csv \
  --samples 0 \
  --decoding-preset standard \
  --notes "Phase 05 data-only locked evaluation"

python training/analyze_errors.py \
  --predictions-csv "reports/runs/${RUN_NAME}-locked-test/predictions.csv" \
  --source-csv dataset/metadata_test.csv
```

Success signal:

- better results on code-switched examples
- better results on short clips
- same or better full locked-test WER/CER

### Experiment C: Combine the winning data change with safer decoding

Only run this after Experiment A and Experiment B are measured independently.

Question:

- Are the data gains and decoding gains complementary?

Commands:

```bash
python training/evaluate_checkpoint.py \
  --model-path "outputs/train_runs/$RUN_NAME" \
  --run-name "${RUN_NAME}-locked-test-safe-decode" \
  --run-type phase05_combined_eval \
  --source-csv dataset/metadata_test.csv \
  --samples 0 \
  --decoding-preset phase05_safe_decode_v1 \
  --notes "Phase 05 combined evaluation with data emphasis plus safer decoding"

python training/analyze_errors.py \
  --predictions-csv "reports/runs/${RUN_NAME}-locked-test-safe-decode/predictions.csv" \
  --source-csv dataset/metadata_test.csv
```

## What Each Change Is Trying To Improve

- Safer decoding:
  - targets `repeated_token_hallucination`
  - targets `catastrophic_looping`
  - may also help some `high_cer_moderate_wer` cases

- Code-switch boost:
  - targets higher average WER on code-switched references
  - gives more pressure on French/English mixed tokens

- Short-clip boost:
  - targets weak very short utterances
  - may reduce the omission-style failures on tiny clips

## Rules

- Compare one major variable at a time first.
- Do not claim a win from `test_head_20` alone.
- Use `reports/experiment_history.csv` plus the locked-test Phase 4 error analysis to decide what actually improved.
