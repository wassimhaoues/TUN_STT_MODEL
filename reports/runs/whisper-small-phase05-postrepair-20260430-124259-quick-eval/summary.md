# Checkpoint Evaluation Report: whisper-small-phase05-postrepair-20260430-124259-quick-eval

This report records a fine-tuned checkpoint evaluation against a tracked manifest.
Use it to compare quick and locked test performance against the raw baseline.

## Run Details

- Run name: `whisper-small-phase05-postrepair-20260430-124259-quick-eval`
- Run type: `phase05_postrepair_eval`
- Model path: `outputs/train_runs/whisper-small-phase05-postrepair-20260430-124259`
- Date: `2026-04-30T15:09:48.914121+01:00`
- Dataset source: `dataset/metadata_test.csv`
- Evaluation scope: `test_head_20`
- Sample count: `20`
- Device: `cuda:0`
- Language: `arabic`
- Task: `transcribe`
- Metric normalization: `v1`

## Decoding Policy

- Decoding preset: `standard`
- Generation max length: `225`
- Generation beams: `1`
- Length penalty: `1.0`
- No-repeat ngram size: `0`
- Repetition penalty: `1.0`

## Metrics

- WER: `0.338462`
- CER: `0.140874`

## Notes

Phase 05 post-repair quick eval with standard decoding
