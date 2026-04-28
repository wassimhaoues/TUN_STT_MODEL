# Phase 03 Full Fine-Tune Report: whisper-small-phase03-full-20260428-001840

This report records the first full Whisper Small fine-tuning run on the complete train and validation manifests.
Its job is to answer whether straightforward supervised fine-tuning already beats the frozen baseline.

## Run Details

- Run name: `whisper-small-phase03-full-20260428-001840`
- Run type: `phase03_full_finetune`
- Model: `openai/whisper-small`
- Created at: `2026-04-28T00:18:59+01:00`
- Git commit: `ead7d1e676d2a3587141119a49f739d5249bd566`
- Git dirty at launch: `True`
- Output dir: `/home/coworky/Study/Deeplearning/TUN_STT_MODEL/outputs/train_runs/whisper-small-phase03-full-20260428-001840`
- Exported best-model path: `/home/coworky/Study/Deeplearning/TUN_STT_MODEL/outputs/train_runs/whisper-small-phase03-full-20260428-001840`
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

- Train rows: `18648` from `18648` (hours `27.660`, duration `0.500` to `29.863` sec)
- Valid rows: `1036` from `1036` (hours `1.564`, duration `0.528` to `27.114` sec)

## Training Metrics

- epoch: `3.000000`
- total_flos: `16144621635502080000.000000`
- train_loss: `2.157858`
- train_runtime: `8278.807600`
- train_samples_per_second: `6.757000`
- train_steps_per_second: `0.423000`

## Validation Metrics

- epoch: `3.000000`
- eval_cer: `0.213782`
- eval_loss: `0.539906`
- eval_runtime: `157.100100`
- eval_samples_per_second: `6.595000`
- eval_steps_per_second: `1.649000`
- eval_wer: `0.435870`

## Best Checkpoint

`/home/coworky/Study/Deeplearning/TUN_STT_MODEL/outputs/train_runs/whisper-small-phase03-full-20260428-001840/checkpoint-3000`

## Notes

Phase 03 first full fine-tune on full normalized train/valid split
