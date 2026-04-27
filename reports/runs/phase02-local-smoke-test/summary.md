# Phase 02 Smoke Training Report: phase02-local-smoke-test

This report records the Phase 02 end-to-end Whisper Small smoke fine-tuning run.
Its job is to prove that training, checkpointing, and validation all work reliably on this project setup.

## Run Details

- Run name: `phase02-local-smoke-test`
- Run type: `phase02_smoke_train`
- Model: `openai/whisper-small`
- Created at: `2026-04-27T23:34:39+01:00`
- Git commit: `f620e080c1ca9b3bd3895c218e1d66d03cc4b68d`
- Git dirty at launch: `True`
- Output dir: `/home/coworky/Study/Deeplearning/TUN_STT_MODEL/outputs/train_runs/phase02-local-smoke-test`
- Train manifest: `/home/coworky/Study/Deeplearning/TUN_STT_MODEL/dataset/metadata_train.csv`
- Valid manifest: `/home/coworky/Study/Deeplearning/TUN_STT_MODEL/dataset/metadata_valid.csv`
- Normalization version: `v1`
- Language: `arabic`
- Task: `transcribe`

## Hardware and Precision

- Device: `cuda:0`
- GPU: `NVIDIA GeForce RTX 5070 Laptop GPU`
- GPU memory (GB): `7.53`
- CPU count: `24`
- RAM (GB): `15.32`
- Precision mode: `bf16`
- Torch: `2.10.0+cu130`
- Transformers: `5.6.2`
- Datasets: `4.8.5`

## Dataset Subsets

- Train rows: `8` from `18648` (hours `0.007`, duration `0.961` to `5.443` sec)
- Valid rows: `4` from `1036` (hours `0.005`, duration `3.191` to `5.660` sec)

## Training Metrics

- epoch: `0.125000`
- total_flos: `288585400320000.000000`
- train_loss: `1.092838`
- train_runtime: `6.334600`
- train_samples_per_second: `0.158000`
- train_steps_per_second: `0.158000`

## Validation Metrics

- epoch: `0.125000`
- eval_cer: `0.410480`
- eval_loss: `2.464015`
- eval_runtime: `1.364200`
- eval_samples_per_second: `2.932000`
- eval_steps_per_second: `2.932000`
- eval_wer: `0.829268`

## Best Checkpoint

`/home/coworky/Study/Deeplearning/TUN_STT_MODEL/outputs/train_runs/phase02-local-smoke-test/checkpoint-1`

## Notes

tiny local validation
