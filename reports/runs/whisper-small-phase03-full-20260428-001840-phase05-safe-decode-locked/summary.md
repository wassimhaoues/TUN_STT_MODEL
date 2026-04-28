# Checkpoint Evaluation Report: whisper-small-phase03-full-20260428-001840-phase05-safe-decode-locked

This report records a fine-tuned checkpoint evaluation against a tracked manifest.
Use it to compare quick and locked test performance against the raw baseline.

## Run Details

- Run name: `whisper-small-phase03-full-20260428-001840-phase05-safe-decode-locked`
- Run type: `phase05_decode_only`
- Model path: `outputs/train_runs/whisper-small-phase03-full-20260428-001840`
- Date: `2026-04-28T14:21:07.875577+01:00`
- Dataset source: `dataset/metadata_test.csv`
- Evaluation scope: `test_full`
- Sample count: `1036`
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

- WER: `0.382745`
- CER: `0.174580`

## Notes

Phase 05 decode-only locked evaluation with safer decoding
