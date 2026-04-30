# Phase 02 Smoke Training Report: whisper-small-phase05-postrepair-20260430-124259

This report records the tracked training run `phase05_postrepair_clean_retrain`.
Use it to compare this run against the baseline and later fine-tuning iterations.

## Run Details

- Run name: `whisper-small-phase05-postrepair-20260430-124259`
- Run type: `phase05_postrepair_clean_retrain`
- Model: `openai/whisper-small`
- Created at: `2026-04-30T12:44:03+01:00`
- Git commit: `a441800114be955ddca6688f41c93aa030a7336e`
- Git dirty at launch: `True`
- Output dir: `/home/coworky/Study/Deeplearning/TUN_STT_MODEL/outputs/train_runs/whisper-small-phase05-postrepair-20260430-124259`
- Exported best-model path: `/home/coworky/Study/Deeplearning/TUN_STT_MODEL/outputs/train_runs/whisper-small-phase05-postrepair-20260430-124259`
- Train manifest: `dataset/metadata_train.csv`
- Valid manifest: `dataset/metadata_valid.csv`
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

## Decoding Policy

- Decoding preset: `standard`
- Generation max length: `225`
- Generation beams: `1`
- Length penalty: `1.0`
- No-repeat ngram size: `0`
- Repetition penalty: `1.0`

## Dataset Subsets

- Train rows: `18639` from `18639` (hours `27.693`, duration `0.500` to `29.863` sec)
- Valid rows: `1036` from `1036` (hours `1.509`, duration `0.540` to `27.406` sec)

## Training Metrics

- epoch: `3.000000`
- total_flos: `16136829829693440000.000000`
- train_loss: `2.123827`
- train_runtime: `8313.857400`
- train_samples_per_second: `6.726000`
- train_steps_per_second: `0.420000`

## Validation Metrics

- epoch: `3.000000`
- eval_cer: `0.189051`
- eval_loss: `0.540120`
- eval_runtime: `164.639700`
- eval_samples_per_second: `6.293000`
- eval_steps_per_second: `1.573000`
- eval_wer: `0.391625`

## Best Checkpoint

`/home/coworky/Study/Deeplearning/TUN_STT_MODEL/outputs/train_runs/whisper-small-phase05-postrepair-20260430-124259/checkpoint-2750`

## Notes

Phase 05 post-repair retrain on repaired default train/valid splits
