# Phase 05 Experiment B Preparation

This prep step isolates training-data cleanup from benchmark corrections.

## Low-Volume Thresholds

- Low-volume RMS threshold: `-36.228 dBFS` (training-split bottom decile)
- Very-low-volume RMS threshold: `-64.369 dBFS` (training-split bottom 3%)

## Split Counts Below Threshold

- Train low-volume rows: `1865`
- Train very-low-volume rows: `560`
- Valid low-volume rows: `103`
- Test low-volume rows: `107`

## Outputs

- Training audit sheet: `dataset/audits/phase05_experiment_b_training_audit.csv`
- Benchmark review sheet: `dataset/audits/phase05_benchmark_reference_audit.csv`
- Full loudness inventory: `reports/phase05_preparation/audio_inventory_all_splits.csv`
- Experiment A freeze: `reports/phase05_preparation/experiment_a_freeze.md`

## Rules

- Use the training audit sheet to decide what may be boosted in Experiment B.
- Keep validation and test corrections separate in the benchmark review sheet. Do not feed them into the training manifest builder.
- Only mark `audio_action=gain_normalize` when the speech is still intelligible and the reference is trustworthy.
- Mark unusable rows for exclusion instead of boosting noise-heavy or transcript-unclear clips.

## Current Candidate Counts

- Training audit candidates: `180`
- Benchmark review candidates: `27`
- Stale manual-review ids skipped: `9`
- Stale error-analysis ids skipped: `720`
