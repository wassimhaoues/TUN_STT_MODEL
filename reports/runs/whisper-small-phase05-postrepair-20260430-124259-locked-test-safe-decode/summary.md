# Checkpoint Evaluation Report: whisper-small-phase05-postrepair-20260430-124259-locked-test-safe-decode

This report records a fine-tuned checkpoint evaluation against a tracked manifest.
Use it to compare quick and locked test performance against the raw baseline.

## Run Details

- Run name: `whisper-small-phase05-postrepair-20260430-124259-locked-test-safe-decode`
- Run type: `phase05_postrepair_combined_eval`
- Model path: `outputs/train_runs/whisper-small-phase05-postrepair-20260430-124259`
- Date: `2026-04-30T15:19:15.631085+01:00`
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

- WER: `0.376286`
- CER: `0.172595`

## Notes

Phase 05 post-repair locked eval with safe decoding
