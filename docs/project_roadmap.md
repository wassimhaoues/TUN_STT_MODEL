# Tunisian Derja Whisper Small Roadmap

## Purpose

This document is the execution plan for taking this repository from a raw Whisper Small baseline to a documented, reproducible Tunisian Derja ASR project with measurable improvements over time.

The roadmap is tailored to the current project state:

- Base model: `openai/whisper-small`
- Current baseline report: `reports/runs/whisper-small-baseline-initial/summary.md`
- Baseline metrics on `test_head_20`: `WER 1.391489`, `CER 0.879585`
- Dataset size today:
    - Train: `18,648` rows, about `27.66 h`
    - Valid: `1,036` rows, about `1.56 h`
    - Test: `1,036` rows, about `1.54 h`
- Hardware target:
    - GPU: `RTX 5070 Laptop GPU`, about `8 GB VRAM`
    - RAM: `15 GB`
    - CPU: `Intel i7-14650HX`

This plan follows current guidance from the Whisper paper, Hugging Face Whisper fine-tuning guidance, Hugging Face `datasets` audio processing docs, Hugging Face `Trainer` docs, and PyTorch mixed-precision docs. See the references section at the end.

## Non-Negotiable Project Rules

These rules apply across all phases:

1. Never touch the test split while making training decisions.
    - `metadata_test.csv` is for locked evaluation only.
    - Fast iteration may use `test_head_20` for continuity with the baseline, but final claims must use the full locked test split.

2. Every experiment must be reproducible.
    - Save config, seed, model checkpoint path, evaluation scope, commit SHA, and notes.
    - Every training or evaluation run must leave a record in `reports/`.

3. Text normalization must be explicit, versioned, and stable.
    - No silent normalization changes between runs.
    - When normalization policy changes, it must be treated as a new experiment generation.

4. Data preparation must preserve traceability.
    - Each training row must be traceable back to source audio and source text.
    - Removed rows must stay documented with reasons.

5. Each phase must answer a real question.
    - Example: "Does stable text normalization improve WER?"
    - Example: "Does full fine-tuning outperform raw Whisper Small on the locked test split?"

## Branching Strategy

Each phase gets its own branch. Merge to `main` only when the phase exit criteria are met.

Recommended branch names:

- `phase-00-baseline-freeze`
- `phase-01-data-contract-and-normalization`
- `phase-02-training-pipeline-smoke`
- `phase-03-first-full-finetune`
- `phase-04-evaluation-and-error-analysis`
- `phase-05-targeted-improvements`
- `phase-06-packaging-and-release`
- `phase-07-final-benchmark-and-closeout`

Branch rules:

- One branch = one primary objective
- No mixed-purpose branches
- Every branch must update docs and reports if behavior or results change
- Every merged phase must leave the repo in a runnable state

## Global Technical Strategy

### Model and training approach

- Start with full fine-tuning of `openai/whisper-small`
- Use multilingual Whisper settings with explicit `language="arabic"` and `task="transcribe"`
- Train with mini-batches, not single-example updates
- Use `bf16` on this GPU
- Use gradient checkpointing to reduce memory pressure
- Start with:
    - `per_device_train_batch_size=4`
    - `gradient_accumulation_steps=4`
    - `per_device_eval_batch_size=4`
    - `group_by_length=True`
    - `num_train_epochs=3`
    - `learning_rate=1e-5`
    - `dataloader_num_workers=2` or `4`

### Audio policy

- Convert all training/eval audio to mono
- Standardize to `16 kHz` for Whisper inputs
- Track clip duration stats before every major training run
- Keep a hard rule that any training clip above Whisper-friendly duration must be split or excluded with a documented reason

### Evaluation policy

Track two evaluation tiers:

- Quick benchmark:
    - `test_head_20`
    - Used for continuity and fast iteration
- Locked benchmark:
    - Full `metadata_test.csv`
    - Used for actual model comparison and release decisions

Core metrics:

- `WER`
- `CER`

Supporting analysis:

- Per-sample predictions CSV
- Error buckets for code-switching, named entities, numerals, repetitions, hallucinations, and dropped phrases

## Phase Plan

## Phase 00: Baseline Freeze

Branch: `phase-00-baseline-freeze`

Goal:

- Freeze the starting point so all future gains can be measured honestly.

Tasks:

- Keep the current baseline report in `reports/`
- Confirm the baseline script is stable and reproducible
- Add a docs index if needed so roadmap and reports are easy to discover
- Record hardware assumptions and starting metrics in docs

Exit criteria:

- Baseline script runs and writes tracked reports
- The repo has a documented starting benchmark
- The team agrees that this is the fixed point of comparison

What this phase tells us:

- Raw Whisper Small on this dataset is not good enough yet
- We now have a real benchmark to beat

## Phase 01: Data Contract and Text Normalization

Branch: `phase-01-data-contract-and-normalization`

Goal:

- Define exactly what “correct transcription text” means for this project.

Tasks:

- Write a normalization policy document for Tunisian Derja transcripts
- Decide and implement rules for:
    - whitespace normalization
    - repeated spaces
    - Arabic punctuation policy
    - Arabic letter variants if any are to be normalized
    - tatweel and obvious decorative characters
    - numbers: keep as spoken form vs digits
    - Latin/French/English code-switched words
    - casing for Latin-script words
    - bracketed noise or non-speech markers
    - empty or near-empty text
- Create a reusable normalization module used by:
    - dataset prep
    - training labels
    - evaluation references
    - metric computation
- Add unit tests with real examples from the dataset
- Rebuild metadata files only if the policy requires it, and record the change

Important caution:

- Do not over-normalize away meaningful Tunisian/French code-switching
- If a token is genuinely spoken, it should usually remain in the label space

Exit criteria:

- A single normalization function exists and is tested
- The exact policy is documented in Markdown
- Metrics can be recomputed consistently under the same text rules

What this phase tells us:

- Whether label inconsistency is one of the main reasons baseline quality looks bad

## Phase 02: Training Pipeline Smoke Test

Branch: `phase-02-training-pipeline-smoke`

Goal:

- Build the full training path end to end on a small subset before spending hours on full runs.

Tasks:

- Implement `training/train_whisper_small.py`
- Implement dataset loading from the current CSV manifests
- Build a Whisper-compatible data pipeline:
    - audio load
    - mono conversion
    - resampling to `16 kHz`
    - feature extraction
    - label tokenization
- Implement a proper data collator for Whisper seq2seq training
- Add config support for:
    - output dir
    - seed
    - train/valid subset size
    - batch size
    - grad accumulation
    - learning rate
    - epochs
    - eval/save/logging steps
- Use a tiny smoke subset first, for example:
    - `1,000` train rows
    - `200` valid rows
- Ensure checkpoints, logs, and evaluation reports are produced

Recommended default training config for this machine:

- `bf16=True`
- `gradient_checkpointing=True`
- `per_device_train_batch_size=4`
- `gradient_accumulation_steps=4`
- `per_device_eval_batch_size=4`
- `group_by_length=True`
- `predict_with_generate=True`
- `save_total_limit` to prevent disk bloat

Exit criteria:

- A short run completes without OOM
- A checkpoint is saved
- Validation metrics are computed
- Run metadata lands in `reports/`

What this phase tells us:

- Whether the project can train reliably on the available laptop hardware

## Phase 03: First Full Fine-Tune

Branch: `phase-03-first-full-finetune`

Goal:

- Run the first real fine-tuning job on the full train/valid split and compare it against baseline.

Tasks:

- Train on full `metadata_train.csv`
- Validate on full `metadata_valid.csv`
- Save:
    - training config
    - best checkpoint
    - final checkpoint
    - validation metric history
- Define the selection rule:
    - choose best model by validation `WER`
- Run quick evaluation on `test_head_20`
- Run locked evaluation on full `metadata_test.csv`
- Record both in `reports/`

Exit criteria:

- One full fine-tuned checkpoint exists
- It has a report with training settings and both quick/full test metrics
- We can say clearly whether it beats raw Whisper Small

What this phase tells us:

- Whether straightforward supervised fine-tuning already gives a meaningful gain

## Phase 04: Evaluation and Error Analysis

Branch: `phase-04-evaluation-and-error-analysis`

Goal:

- Understand failure modes instead of blindly tuning hyperparameters.

Tasks:

- Implement `training/evaluate_checkpoint.py`
- Produce structured prediction reports on:
    - quick benchmark
    - full locked test split
- Create an error analysis notebook or script that buckets examples by:
    - code-switching with French/English
    - Arabic-only speech
    - short clips
    - long clips
    - high CER but moderate WER
    - repeated-token hallucination
    - major omissions
- Sample and annotate a small fixed error-analysis set manually

Exit criteria:

- A written error summary exists
- We know the top 3 to 5 failure modes
- The next optimization steps are based on evidence, not intuition

What this phase tells us:

- Where the model is actually failing: normalization, acoustics, code-switching, duration, or decoding behavior

## Phase 05: Targeted Improvements

Branch: `phase-05-targeted-improvements`

Goal:

- Improve the model using findings from Phase 04, one controlled change at a time.

Possible tracks:

- Data track:
    - remove bad labels
    - fix obvious transcript inconsistencies
    - tighten duration thresholds
    - optionally add segmentation for long utterances
- Text track:
    - refine normalization policy
    - improve handling of Latin-script code-switching
- Training track:
    - tune learning rate
    - tune epochs
    - tune effective batch size
    - tune warmup ratio
    - compare best-vs-last checkpoint behavior
- Decoding track:
    - test generation settings
    - compare constrained language/task settings

Rules for this phase:

- Only one major variable change per experiment family
- Every experiment must declare:
    - what changed
    - why it changed
    - what metric is expected to improve

Exit criteria:

- At least one targeted change beats the Phase 03 model on the locked test set
- Losing changes are also documented so they are not retried blindly

What this phase tells us:

- Which interventions actually matter for Tunisian Derja on this dataset

## Phase 06: Packaging and Release Readiness

Branch: `phase-06-packaging-and-release`

Goal:

- Make the best model usable, reproducible, and shareable.

Tasks:

- Create a clean inference entrypoint
- Add a simple local transcription CLI
- Add a model report template for future checkpoints
- Prepare a Hugging Face model card with:
    - base model
    - training data description
    - evaluation results
    - intended use
    - known limitations
- Prepare a dataset card if the dataset is to be shared
- Decide whether to publish:
    - weights only
    - weights plus processor
    - sample evaluation artifacts

Exit criteria:

- Best checkpoint can be loaded and used outside the training script
- The release artifacts are documented
- Reproduction instructions are complete

What this phase tells us:

- Whether the project is mature enough to be shared or reused

## Phase 07: Final Benchmark and Closeout

Branch: `phase-07-final-benchmark-and-closeout`

Goal:

- Produce the final project result with clear evidence of improvement from start to finish.

Tasks:

- Re-run the baseline if needed under the final normalization/evaluation policy
- Re-run final model evaluation on:
    - `test_head_20`
    - full locked test split
- Build a comparison table:
    - raw Whisper Small
    - first fine-tuned model
    - best final model
- Write a final summary including:
    - what worked
    - what did not work
    - where the model still fails
    - what the next project should try

Exit criteria:

- One final benchmark report exists
- Improvement over baseline is quantified and reproducible
- Open risks and future work are documented

What this phase tells us:

- Whether the project achieved a meaningful, honest gain over the starting point

## Cross-Phase Deliverables

These deliverables should grow throughout the project:

- `reports/experiment_history.csv`
- run-level Markdown summaries
- per-run prediction CSV files
- normalization policy doc
- training config files
- evaluation scripts
- error analysis notes
- final model card and possibly dataset card

## Recommended File Additions Over Time

- `docs/normalization_policy.md`
- `training/train_whisper_small.py`
- `training/evaluate_checkpoint.py`
- `training/configs/`
- `training/utils/`
- `docs/error_analysis.md`
- `docs/final_results.md`

## Decision Log We Already Know

- We are starting from Whisper Small, not Tiny or Medium
- We will keep the current quick benchmark for continuity
- We will also use the full locked test set for real comparisons
- We will train in mini-batches
- We will use the current laptop GPU as the primary training environment
- We will preserve code-switching instead of pretending the data is pure Arabic

## Risks to Manage Early

- Transcript inconsistency may dominate metric noise
- Code-switching may confuse both normalization and decoding
- A small VRAM budget may force careful batch sizing
- A weak validation protocol could lead to false optimism
- If the test set influences normalization or tuning choices, final claims become unreliable

## References

1. Whisper paper: OpenAI, "Robust Speech Recognition via Large-Scale Weak Supervision"
   https://openreview.net/pdf?id=Xr12kpEP3G

2. Hugging Face blog, "Fine-Tune Whisper For Multilingual ASR with Transformers"
   https://huggingface.co/blog/fine-tune-whisper

3. Hugging Face Transformers Whisper docs
   https://huggingface.co/docs/transformers/en/model_doc/whisper

4. Hugging Face Datasets audio processing docs
   https://huggingface.co/docs/datasets/audio_process

5. Hugging Face Trainer docs
   https://huggingface.co/docs/transformers/main_classes/trainer

6. PyTorch Automatic Mixed Precision docs
   https://docs.pytorch.org/docs/stable/accelerator/amp.html

7. Hugging Face model card docs
   https://huggingface.co/docs/hub/model-cards

8. Hugging Face dataset card docs
   https://huggingface.co/docs/hub/en/datasets-cards

9. TunSwitch dataset / code-switched Tunisian Arabic ASR resource
   https://zenodo.org/records/8342762
