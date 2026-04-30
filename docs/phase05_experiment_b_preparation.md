# Phase 05 Experiment B Preparation

This step prepares a clean, isolated data audit before the first data-only Experiment B run.

## Goals

- freeze the Experiment A decode-only win as the current best inference policy
- keep Experiment B focused on data changes only
- prepare a fill-in audit sheet for training rows that may be boosted
- review low-volume audio systematically before duplicating noisy examples
- protect validation and test from accidental training-side cleanup

## Command

```bash
source train-env/bin/activate
python dataset/prepare_phase05_experiment_b.py
```

## Files Produced

- `reports/phase05_preparation/experiment_a_freeze.md`
- `reports/phase05_preparation/summary.md`
- `reports/phase05_preparation/audio_inventory_all_splits.csv`
- `dataset/audits/phase05_experiment_b_training_audit.csv`
- `dataset/audits/phase05_benchmark_reference_audit.csv`

## How To Use The Training Audit Sheet

Fill these columns by hand:

- `audit_status`
  Use one of: `reviewed`, `unclear_audio`, `wrong_transcript`, `skip`
- `transcript_action`
  Use one of: `keep`, `fix_text`, `exclude`
- `audio_action`
  Use one of: `keep_raw`, `gain_normalize`, `exclude`
- `keep_for_phase05_boost`
  Use `yes` or `no`
- `corrected_text`
  Fill only when `transcript_action=fix_text`
- `notes`
  Short reason, especially when excluding a row

## How To Use The Benchmark Review Sheet

This file is intentionally separate.

- It is for validation/test reference review only.
- Do not feed it into the Experiment B training manifest builder.
- Use it to capture benchmark-side label problems or unclear audio that should be tracked separately.

Fill these columns by hand:

- `benchmark_audit_status`
  Use one of: `reviewed`, `unclear_audio`, `wrong_reference`, `model_better_than_reference`
- `reference_action`
  Use one of: `keep`, `fix_reference`, `flag_only`
- `corrected_reference`
  Fill only when `reference_action=fix_reference`
- `notes`
  Add why the sample was flagged

## Decision Rule For Quiet Audio

Use `gain_normalize` only when all of these are true:

- speech is still intelligible
- the transcript is trustworthy
- the file is quiet enough to matter
- boosting is more likely to reveal speech than to amplify mostly noise

If a clip is quiet and unclear, prefer `exclude` over boosting noise.

## Isolation Rule

The first Experiment B run should remain data-only:

- train with the cleaned or filtered training manifest
- keep validation unchanged
- evaluate with `--decoding-preset standard`

Only after the data-only result is measured should you combine the new checkpoint with the frozen safe decode preset from Experiment A.
