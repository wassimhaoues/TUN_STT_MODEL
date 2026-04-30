# Phase 05 Data Strategy: phase05-codeswitch-short-boost-audited-v1

This strategy boosts targeted training examples without changing the validation set.

## Inputs

- Train manifest: `/home/coworky/Study/Deeplearning/TUN_STT_MODEL/dataset/metadata_train.csv`
- Valid manifest: `/home/coworky/Study/Deeplearning/TUN_STT_MODEL/dataset/metadata_valid.csv`
- Training audit CSV: `/home/coworky/Study/Deeplearning/TUN_STT_MODEL/dataset/audits/phase05_experiment_b_training_audit.csv`

## Strategy

- Code-switch boost factor: `2`
- Short-clip boost factor: `2`
- Short-clip threshold: `3.000` seconds

## Counts

- Train rows in: `18639`
- Train rows out: `33224`
- Valid rows in: `1036`
- Valid rows out: `1036`
- Excluded train rows from audit: `0`
- Corrected train rows from audit: `0`
- Boost-approved audit rows: `0`
- Gain-normalize flagged rows: `0`
- Code-switched train rows: `10708`
- Short train rows: `6022`
- Code-switched and short train rows: `2037`

## Outputs

- Output train manifest: `/home/coworky/Study/Deeplearning/TUN_STT_MODEL/dataset/phase05_manifests/phase05-codeswitch-short-boost-audited-v1/metadata_train.csv`
- Output valid manifest: `/home/coworky/Study/Deeplearning/TUN_STT_MODEL/dataset/phase05_manifests/phase05-codeswitch-short-boost-audited-v1/metadata_valid.csv`
