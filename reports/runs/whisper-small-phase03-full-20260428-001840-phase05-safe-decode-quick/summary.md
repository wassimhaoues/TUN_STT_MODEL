# Checkpoint Evaluation Report: whisper-small-phase03-full-20260428-001840-phase05-safe-decode-quick

This report records a fine-tuned checkpoint evaluation against a tracked manifest.
Use it to compare quick and locked test performance against the raw baseline.

## Run Details

- Run name: `whisper-small-phase03-full-20260428-001840-phase05-safe-decode-quick`
- Run type: `phase05_decode_only`
- Model path: `outputs/train_runs/whisper-small-phase03-full-20260428-001840`
- Date: `2026-04-28T14:15:49.810844+01:00`
- Dataset source: `dataset/metadata_test.csv`
- Evaluation scope: `test_head_20`
- Sample count: `20`
- Device: `cuda:0`
- Language: `arabic`
- Task: `transcribe`
- Metric normalization: `v1`

## Decoding Policy

- Decoding preset: `phase05_safe_decode_v1`
- Generation max length: `225`
- Generation beams: `3`
- Length penalty: `1.0`
- No-repeat ngram size: `3`
- Repetition penalty: `1.1`

## Metrics

- WER: `0.344681`
- CER: `0.136364`

## Notes

Phase 05 decode-only quick evaluation with safer decoding
