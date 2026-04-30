# Checkpoint Evaluation Report: whisper-small-phase05-postrepair-20260430-124259-locked-test

This report records a fine-tuned checkpoint evaluation against a tracked manifest.
Use it to compare quick and locked test performance against the raw baseline.

## Run Details

- Run name: `whisper-small-phase05-postrepair-20260430-124259-locked-test`
- Run type: `phase05_postrepair_eval`
- Model path: `outputs/train_runs/whisper-small-phase05-postrepair-20260430-124259`
- Date: `2026-04-30T15:10:13.631976+01:00`
- Dataset source: `dataset/metadata_test.csv`
- Evaluation scope: `test_full`
- Sample count: `1036`
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

- WER: `0.371431`
- CER: `0.179962`

## Notes

Phase 05 post-repair locked eval with standard decoding
