# Phase 05 Experiment A Freeze

This file freezes the current best decode-only result before Experiment B.

## Current Winner

- Frozen run: `whisper-small-phase03-full-20260428-001840-phase05-safe-decode-locked`
- Frozen decoding preset: `phase05_safe_decode_v1`
- Status: use this as the current best inference policy

## Locked Test Comparison

- Reference locked run: `whisper-small-phase03-full-20260428-001840-locked-test`
- Reference WER: `0.489161`
- Reference CER: `0.256869`
- Frozen WER: `0.382745`
- Frozen CER: `0.174580`
- WER delta: `-0.106416`
- CER delta: `-0.082289`

## Experiment B Isolation Rule

Train and evaluate the first data-only Experiment B run with `--decoding-preset standard` so the data effect stays isolated.

Only after the data-only result is measured should the frozen safe decode preset be combined with the new checkpoint.
