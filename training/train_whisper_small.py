from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataset.text_normalization import NORMALIZATION_VERSION, normalize_transcript  # noqa: E402

ProcessorLike = Any
EvalPredictionLike = Any

DATASET_DIR = ROOT_DIR / "dataset"
WAV_DIR = DATASET_DIR / "extracted_wavs"
REPORTS_DIR = ROOT_DIR / "reports"
DEFAULT_TRAIN_CSV = DATASET_DIR / "metadata_train.csv"
DEFAULT_VALID_CSV = DATASET_DIR / "metadata_valid.csv"
DEFAULT_MODEL_NAME = "openai/whisper-small"
DEFAULT_RUN_TYPE = "phase02_smoke_train"
DEFAULT_PHASE_LABEL = "Phase 02 Smoke Training"
DEFAULT_LANGUAGE = "arabic"
DEFAULT_TASK = "transcribe"
DEFAULT_TRAIN_SAMPLES = 1000
DEFAULT_VALID_SAMPLES = 200
DEFAULT_TRAIN_BATCH_SIZE = 4
DEFAULT_EVAL_BATCH_SIZE = 4
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 4
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_NUM_TRAIN_EPOCHS = 1.0
DEFAULT_MAX_STEPS = 60
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_EVAL_STEPS = 20
DEFAULT_SAVE_STEPS = 20
DEFAULT_LOGGING_STEPS = 5
DEFAULT_SAVE_TOTAL_LIMIT = 2
DEFAULT_DATALOADER_NUM_WORKERS = 2
DEFAULT_GENERATION_MAX_LENGTH = 225
DEFAULT_MAX_DURATION_SECONDS = 30.0
TARGET_SAMPLING_RATE = 16000
EXPERIMENT_HISTORY_COLUMNS = [
    "run_name",
    "run_type",
    "model_name",
    "eval_scope",
    "n_samples",
    "wer",
    "cer",
    "device",
    "language",
    "task",
    "source_csv",
    "created_at",
    "notes",
]


@dataclass(frozen=True)
class ManifestRow:
    id: str
    text: str
    duration: float
    text_raw: str
    normalization_changed: bool
    normalization_version: str
    audio_path: str


@dataclass(frozen=True)
class DatasetProfile:
    split_name: str
    source_csv: str
    selected_rows: int
    requested_rows: int
    total_available_rows: int
    total_hours: float
    min_duration: float
    median_duration: float
    max_duration: float
    normalization_version: str
    sample_ids: list[str]


@dataclass(frozen=True)
class EnvironmentSnapshot:
    device: str
    gpu_name: str
    gpu_total_memory_gb: float
    cpu_count: int
    ram_gb: float
    precision: str
    torch_version: str
    transformers_version: str
    datasets_version: str


@dataclass(frozen=True)
class TrainingConfig:
    run_name: str
    run_type: str
    model_name: str
    train_csv: str
    valid_csv: str
    train_samples: int
    valid_samples: int
    seed: int
    output_dir: str
    reports_dir: str
    language: str
    task: str
    precision: str
    gradient_checkpointing: bool
    group_by_length: bool
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    num_train_epochs: float
    max_steps: int
    warmup_ratio: float
    eval_steps: int
    save_steps: int
    logging_steps: int
    save_total_limit: int
    dataloader_num_workers: int
    generation_max_length: int
    max_duration_seconds: float
    notes: str
    resume_from_checkpoint: str
    created_at: str
    git_commit: str
    git_dirty: bool


@dataclass(frozen=True)
class PrecisionPlan:
    label: str
    bf16: bool
    fp16: bool


@dataclass(frozen=True)
class TrainingRunArtifacts:
    run_dir: Path
    summary_path: Path
    config_path: Path
    environment_path: Path
    train_manifest_path: Path
    valid_manifest_path: Path
    dataset_profile_path: Path
    train_metrics_path: Path
    eval_metrics_path: Path


@dataclass(frozen=True)
class TrainingRunResult:
    config: TrainingConfig
    environment: EnvironmentSnapshot
    train_profile: DatasetProfile
    valid_profile: DatasetProfile
    train_metrics: dict[str, float]
    eval_metrics: dict[str, float]
    best_checkpoint: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a reproducible Phase 02 Whisper Small smoke fine-tuning job."
    )
    parser.add_argument(
        "--run-name",
        help="Optional run name. Defaults to a timestamped smoke name.",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Optional notes to add to the tracked report.",
    )
    parser.add_argument(
        "--run-type",
        default=DEFAULT_RUN_TYPE,
        help=(f"Tracked run type label. Defaults to {DEFAULT_RUN_TYPE}."),
    )
    parser.add_argument(
        "--train-csv",
        default=str(DEFAULT_TRAIN_CSV),
        help="Training manifest CSV path.",
    )
    parser.add_argument(
        "--valid-csv",
        default=str(DEFAULT_VALID_CSV),
        help="Validation manifest CSV path.",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=DEFAULT_TRAIN_SAMPLES,
        help=f"Rows to sample from the training manifest. Defaults to {DEFAULT_TRAIN_SAMPLES}.",
    )
    parser.add_argument(
        "--valid-samples",
        type=int,
        default=DEFAULT_VALID_SAMPLES,
        help=f"Rows to sample from the validation manifest. Defaults to {DEFAULT_VALID_SAMPLES}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subset selection and training.",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional output directory for checkpoints and trainer state.",
    )
    parser.add_argument(
        "--reports-dir",
        default=str(REPORTS_DIR),
        help="Tracked reports directory.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Base Whisper checkpoint.",
    )
    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help="Whisper generation language.",
    )
    parser.add_argument("--task", default=DEFAULT_TASK, help="Whisper generation task.")
    parser.add_argument(
        "--precision",
        choices=["auto", "bf16", "fp16", "fp32"],
        default="auto",
        help="Mixed-precision mode. Defaults to auto.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable gradient checkpointing.",
    )
    parser.add_argument(
        "--group-by-length",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable length-grouped batching.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=DEFAULT_TRAIN_BATCH_SIZE,
        help=f"Per-device train batch size. Defaults to {DEFAULT_TRAIN_BATCH_SIZE}.",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=DEFAULT_EVAL_BATCH_SIZE,
        help=f"Per-device eval batch size. Defaults to {DEFAULT_EVAL_BATCH_SIZE}.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=DEFAULT_GRADIENT_ACCUMULATION_STEPS,
        help=(f"Gradient accumulation steps. Defaults to {DEFAULT_GRADIENT_ACCUMULATION_STEPS}."),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"Learning rate. Defaults to {DEFAULT_LEARNING_RATE}.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=DEFAULT_NUM_TRAIN_EPOCHS,
        help=f"Training epochs. Defaults to {DEFAULT_NUM_TRAIN_EPOCHS}.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=("Maximum optimization steps for the smoke run. Use -1 to rely fully on epochs."),
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=DEFAULT_WARMUP_RATIO,
        help=f"Warmup ratio. Defaults to {DEFAULT_WARMUP_RATIO}.",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=DEFAULT_EVAL_STEPS,
        help=f"Evaluation interval in steps. Defaults to {DEFAULT_EVAL_STEPS}.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=DEFAULT_SAVE_STEPS,
        help=f"Checkpoint save interval in steps. Defaults to {DEFAULT_SAVE_STEPS}.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=DEFAULT_LOGGING_STEPS,
        help=f"Logging interval in steps. Defaults to {DEFAULT_LOGGING_STEPS}.",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=DEFAULT_SAVE_TOTAL_LIMIT,
        help=f"Maximum checkpoints to keep. Defaults to {DEFAULT_SAVE_TOTAL_LIMIT}.",
    )
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=DEFAULT_DATALOADER_NUM_WORKERS,
        help=(f"Dataloader worker count. Defaults to {DEFAULT_DATALOADER_NUM_WORKERS}."),
    )
    parser.add_argument(
        "--generation-max-length",
        type=int,
        default=DEFAULT_GENERATION_MAX_LENGTH,
        help=(
            "Generation max length for eval-time decoding. "
            f"Defaults to {DEFAULT_GENERATION_MAX_LENGTH}."
        ),
    )
    parser.add_argument(
        "--max-duration-seconds",
        type=float,
        default=DEFAULT_MAX_DURATION_SECONDS,
        help=(
            "Exclude rows longer than this duration before subset selection. "
            f"Defaults to {DEFAULT_MAX_DURATION_SECONDS}."
        ),
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default="",
        help="Optional checkpoint path to resume from.",
    )
    return parser.parse_args()


def sanitize_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() else "-" for char in value.lower())
    return "-".join(part for part in cleaned.split("-") if part) or "run"


def build_run_name(model_name: str, created_at: datetime) -> str:
    model_slug = sanitize_name(model_name.split("/")[-1])
    timestamp = created_at.strftime("%Y%m%d-%H%M%S")
    return f"{model_slug}-phase02-smoke-{timestamp}"


def validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0.")


def validate_non_negative(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be 0 or greater.")


def validate_training_config(config: TrainingConfig) -> None:
    validate_non_negative("train_samples", config.train_samples)
    validate_non_negative("valid_samples", config.valid_samples)
    validate_positive("per_device_train_batch_size", config.per_device_train_batch_size)
    validate_positive("per_device_eval_batch_size", config.per_device_eval_batch_size)
    validate_positive("gradient_accumulation_steps", config.gradient_accumulation_steps)
    validate_positive("eval_steps", config.eval_steps)
    validate_positive("save_steps", config.save_steps)
    validate_positive("logging_steps", config.logging_steps)
    validate_positive("save_total_limit", config.save_total_limit)
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be greater than 0.")
    if config.num_train_epochs <= 0:
        raise ValueError("num_train_epochs must be greater than 0.")
    if config.max_steps == 0:
        raise ValueError("max_steps cannot be 0. Use -1 for epoch-only training.")
    if config.warmup_ratio < 0 or config.warmup_ratio >= 1:
        raise ValueError("warmup_ratio must be in the range [0, 1).")
    if config.max_duration_seconds <= 0:
        raise ValueError("max_duration_seconds must be greater than 0.")
    if config.precision not in {"auto", "bf16", "fp16", "fp32"}:
        raise ValueError("precision must be one of: auto, bf16, fp16, fp32.")


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def is_git_dirty() -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return bool(result.stdout.strip())


def default_output_dir(run_name: str) -> str:
    return str(ROOT_DIR / "outputs" / "train_runs" / run_name)


def resolve_training_config(args: argparse.Namespace) -> TrainingConfig:
    created_at = datetime.now().astimezone()
    run_name = args.run_name or build_run_name(args.model_name, created_at)

    config = TrainingConfig(
        run_name=run_name,
        run_type=args.run_type,
        model_name=args.model_name,
        train_csv=str(Path(args.train_csv)),
        valid_csv=str(Path(args.valid_csv)),
        train_samples=args.train_samples,
        valid_samples=args.valid_samples,
        seed=args.seed,
        output_dir=args.output_dir or default_output_dir(run_name),
        reports_dir=str(Path(args.reports_dir)),
        language=args.language,
        task=args.task,
        precision=args.precision,
        gradient_checkpointing=args.gradient_checkpointing,
        group_by_length=args.group_by_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.dataloader_num_workers,
        generation_max_length=args.generation_max_length,
        max_duration_seconds=args.max_duration_seconds,
        notes=args.notes,
        resume_from_checkpoint=args.resume_from_checkpoint,
        created_at=created_at.isoformat(timespec="seconds"),
        git_commit=get_git_commit(),
        git_dirty=is_git_dirty(),
    )
    validate_training_config(config)
    return config


def load_manifest_rows(csv_path: Path) -> list[ManifestRow]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing manifest: {csv_path}")

    rows: list[ManifestRow] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            sample_id = str(raw_row["id"])
            rows.append(
                ManifestRow(
                    id=sample_id,
                    text=str(raw_row["text"]),
                    duration=float(raw_row["duration"]),
                    text_raw=str(raw_row.get("text_raw", raw_row["text"])),
                    normalization_changed=str(raw_row.get("normalization_changed", "False"))
                    .strip()
                    .lower()
                    == "true",
                    normalization_version=str(
                        raw_row.get("normalization_version", NORMALIZATION_VERSION)
                    ),
                    audio_path=str(WAV_DIR / f"{sample_id}.wav"),
                )
            )
    return rows


def select_rows(
    rows: list[ManifestRow],
    sample_size: int,
    seed: int,
    max_duration_seconds: float,
) -> list[ManifestRow]:
    eligible = [
        row
        for row in rows
        if row.text.strip()
        and row.duration <= max_duration_seconds
        and Path(row.audio_path).exists()
    ]
    if sample_size == 0:
        return sorted(eligible, key=lambda row: (row.duration, row.id))

    if len(eligible) < sample_size:
        raise ValueError(
            f"Requested {sample_size} rows, but only {len(eligible)} eligible rows remain "
            f"after filtering by duration <= {max_duration_seconds} and existing audio."
        )

    rng = random.Random(seed)
    indices = list(range(len(eligible)))
    rng.shuffle(indices)
    selected = [eligible[index] for index in indices[:sample_size]]
    return sorted(selected, key=lambda row: (row.duration, row.id))


def build_dataset_profile(
    split_name: str,
    source_csv: Path,
    selected_rows: list[ManifestRow],
    requested_rows: int,
    total_available_rows: int,
) -> DatasetProfile:
    durations = sorted(row.duration for row in selected_rows)
    mid = len(durations) // 2
    median_duration = (
        durations[mid] if len(durations) % 2 else (durations[mid - 1] + durations[mid]) / 2
    )
    return DatasetProfile(
        split_name=split_name,
        source_csv=str(source_csv),
        selected_rows=len(selected_rows),
        requested_rows=requested_rows or len(selected_rows),
        total_available_rows=total_available_rows,
        total_hours=round(sum(durations) / 3600, 3),
        min_duration=round(durations[0], 3),
        median_duration=round(median_duration, 3),
        max_duration=round(durations[-1], 3),
        normalization_version=NORMALIZATION_VERSION,
        sample_ids=[row.id for row in selected_rows[:10]],
    )


def build_history_row(
    config: TrainingConfig,
    environment: EnvironmentSnapshot,
    train_profile: DatasetProfile,
    valid_profile: DatasetProfile,
    eval_metrics: dict[str, float],
    best_checkpoint: str,
) -> dict[str, str]:
    note_segments = [config.notes.strip()] if config.notes.strip() else []
    note_segments.append(f"train_samples={config.train_samples}")
    note_segments.append(f"valid_samples={config.valid_samples}")
    note_segments.append(f"precision={environment.precision}")
    note_segments.append(f"best_checkpoint={best_checkpoint or 'n/a'}")
    return {
        "run_name": config.run_name,
        "run_type": config.run_type,
        "model_name": config.model_name,
        "eval_scope": (
            "valid_full"
            if valid_profile.selected_rows == valid_profile.total_available_rows
            else f"valid_head_{valid_profile.selected_rows}"
        ),
        "n_samples": str(train_profile.selected_rows),
        "wer": format_metric(eval_metrics.get("eval_wer", float("nan"))),
        "cer": format_metric(eval_metrics.get("eval_cer", float("nan"))),
        "device": environment.device,
        "language": config.language,
        "task": config.task,
        "source_csv": f"{config.train_csv}|{config.valid_csv}",
        "created_at": config.created_at,
        "notes": " | ".join(note_segments),
    }


def format_metric(value: float) -> str:
    if value != value:
        return "nan"
    return f"{value:.6f}"


def append_experiment_history(
    reports_dir: Path,
    config: TrainingConfig,
    environment: EnvironmentSnapshot,
    train_profile: DatasetProfile,
    valid_profile: DatasetProfile,
    eval_metrics: dict[str, float],
    best_checkpoint: str,
) -> None:
    history_path = reports_dir / "experiment_history.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not history_path.exists()
    with history_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=EXPERIMENT_HISTORY_COLUMNS,
            lineterminator="\n",
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            build_history_row(
                config,
                environment,
                train_profile,
                valid_profile,
                eval_metrics,
                best_checkpoint,
            )
        )


def ensure_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_selected_manifest(path: Path, rows: list[ManifestRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "text",
                "duration",
                "text_raw",
                "normalization_changed",
                "normalization_version",
                "audio_path",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def build_summary_markdown(result: TrainingRunResult) -> str:
    phase_label = (
        "Phase 03 Full Fine-Tune"
        if result.config.run_type == "phase03_full_finetune"
        else DEFAULT_PHASE_LABEL
    )
    opening_lines = {
        "phase03_full_finetune": [
            (
                "This report records the first full Whisper Small fine-tuning run "
                "on the complete train and validation manifests."
            ),
            (
                "Its job is to answer whether straightforward supervised "
                "fine-tuning already beats the frozen baseline."
            ),
        ],
        DEFAULT_RUN_TYPE: [
            "This report records the Phase 02 end-to-end Whisper Small smoke fine-tuning run.",
            (
                "Its job is to prove that training, checkpointing, and validation "
                "all work reliably on this project setup."
            ),
        ],
    }.get(
        result.config.run_type,
        [
            f"This report records the tracked training run `{result.config.run_type}`.",
            "Use it to compare this run against the baseline and later fine-tuning iterations.",
        ],
    )
    train_metrics_lines = [
        f"- {key}: `{format_metric(value)}`" for key, value in sorted(result.train_metrics.items())
    ]
    eval_metrics_lines = [
        f"- {key}: `{format_metric(value)}`" for key, value in sorted(result.eval_metrics.items())
    ]
    notes = result.config.notes or "No additional notes."

    return "\n".join(
        [
            f"# {phase_label} Report: {result.config.run_name}",
            "",
            *opening_lines,
            "",
            "## Run Details",
            "",
            f"- Run name: `{result.config.run_name}`",
            f"- Run type: `{result.config.run_type}`",
            f"- Model: `{result.config.model_name}`",
            f"- Created at: `{result.config.created_at}`",
            f"- Git commit: `{result.config.git_commit}`",
            f"- Git dirty at launch: `{result.config.git_dirty}`",
            f"- Output dir: `{result.config.output_dir}`",
            f"- Exported best-model path: `{result.config.output_dir}`",
            f"- Train manifest: `{result.config.train_csv}`",
            f"- Valid manifest: `{result.config.valid_csv}`",
            f"- Normalization version: `{NORMALIZATION_VERSION}`",
            f"- Language: `{result.config.language}`",
            f"- Task: `{result.config.task}`",
            "",
            "## Hardware and Precision",
            "",
            f"- Device: `{result.environment.device}`",
            f"- GPU: `{result.environment.gpu_name}`",
            f"- GPU memory (GB): `{result.environment.gpu_total_memory_gb:.2f}`",
            f"- CPU count: `{result.environment.cpu_count}`",
            f"- RAM (GB): `{result.environment.ram_gb:.2f}`",
            f"- Precision mode: `{result.environment.precision}`",
            f"- Torch: `{result.environment.torch_version}`",
            f"- Transformers: `{result.environment.transformers_version}`",
            f"- Datasets: `{result.environment.datasets_version}`",
            "",
            "## Dataset Subsets",
            "",
            (
                f"- Train rows: `{result.train_profile.selected_rows}` "
                f"from `{result.train_profile.total_available_rows}` "
                f"(hours `{result.train_profile.total_hours:.3f}`, "
                f"duration `{result.train_profile.min_duration:.3f}` "
                f"to `{result.train_profile.max_duration:.3f}` sec)"
            ),
            (
                f"- Valid rows: `{result.valid_profile.selected_rows}` "
                f"from `{result.valid_profile.total_available_rows}` "
                f"(hours `{result.valid_profile.total_hours:.3f}`, "
                f"duration `{result.valid_profile.min_duration:.3f}` "
                f"to `{result.valid_profile.max_duration:.3f}` sec)"
            ),
            "",
            "## Training Metrics",
            "",
            *train_metrics_lines,
            "",
            "## Validation Metrics",
            "",
            *eval_metrics_lines,
            "",
            "## Best Checkpoint",
            "",
            f"`{result.best_checkpoint or 'No best checkpoint recorded.'}`",
            "",
            "## Notes",
            "",
            notes,
            "",
        ]
    )


def create_artifacts(run_name: str, reports_dir: Path) -> TrainingRunArtifacts:
    run_dir = reports_dir / "runs" / run_name
    return TrainingRunArtifacts(
        run_dir=run_dir,
        summary_path=run_dir / "summary.md",
        config_path=run_dir / "training_config.json",
        environment_path=run_dir / "environment.json",
        train_manifest_path=run_dir / "selected_train_manifest.csv",
        valid_manifest_path=run_dir / "selected_valid_manifest.csv",
        dataset_profile_path=run_dir / "dataset_profile.json",
        train_metrics_path=run_dir / "train_metrics.json",
        eval_metrics_path=run_dir / "eval_metrics.json",
    )


def save_run_artifacts(
    artifacts: TrainingRunArtifacts,
    config: TrainingConfig,
    environment: EnvironmentSnapshot,
    train_profile: DatasetProfile,
    valid_profile: DatasetProfile,
    train_rows: list[ManifestRow],
    valid_rows: list[ManifestRow],
    train_metrics: dict[str, float] | None = None,
    eval_metrics: dict[str, float] | None = None,
    summary_markdown: str | None = None,
) -> None:
    artifacts.run_dir.mkdir(parents=True, exist_ok=True)
    ensure_json(artifacts.config_path, asdict(config))
    ensure_json(artifacts.environment_path, asdict(environment))
    ensure_json(
        artifacts.dataset_profile_path,
        {
            "train": asdict(train_profile),
            "valid": asdict(valid_profile),
        },
    )
    write_selected_manifest(artifacts.train_manifest_path, train_rows)
    write_selected_manifest(artifacts.valid_manifest_path, valid_rows)
    if train_metrics is not None:
        ensure_json(artifacts.train_metrics_path, train_metrics)
    if eval_metrics is not None:
        ensure_json(artifacts.eval_metrics_path, eval_metrics)
    if summary_markdown is not None:
        artifacts.summary_path.write_text(summary_markdown, encoding="utf-8")


def resolve_precision_plan(
    config_precision: str,
    has_cuda: bool,
    bf16_supported: bool,
) -> PrecisionPlan:
    if config_precision == "bf16":
        return PrecisionPlan(label="bf16", bf16=True, fp16=False)
    if config_precision == "fp16":
        return PrecisionPlan(label="fp16", bf16=False, fp16=True)
    if config_precision == "fp32":
        return PrecisionPlan(label="fp32", bf16=False, fp16=False)
    if has_cuda and bf16_supported:
        return PrecisionPlan(label="bf16", bf16=True, fp16=False)
    if has_cuda:
        return PrecisionPlan(label="fp16", bf16=False, fp16=True)
    return PrecisionPlan(label="fp32", bf16=False, fp16=False)


def load_audio_for_training(wav_path: Path) -> list[float]:
    import soundfile as sf
    from scipy.signal import resample_poly

    audio, sample_rate = sf.read(str(wav_path))
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    if sample_rate != TARGET_SAMPLING_RATE:
        audio = resample_poly(audio, TARGET_SAMPLING_RATE, sample_rate)
    return audio.astype("float32", copy=False).tolist()


def compute_warmup_steps(
    train_rows: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_train_epochs: float,
    max_steps: int,
    warmup_ratio: float,
) -> int:
    if max_steps > 0:
        total_steps = max_steps
    else:
        steps_per_epoch = math.ceil(train_rows / per_device_train_batch_size)
        total_steps = math.ceil((steps_per_epoch * num_train_epochs) / gradient_accumulation_steps)
    return int(total_steps * warmup_ratio)


def make_transform(processor: ProcessorLike):
    def transform(example: dict[str, Any]) -> dict[str, Any]:
        audio_path = example["audio_path"]

        if isinstance(audio_path, list):
            audio_batch = [load_audio_for_training(Path(str(path))) for path in audio_path]
            features: dict[str, Any] = processor.feature_extractor(
                audio_batch,
                sampling_rate=TARGET_SAMPLING_RATE,
                return_attention_mask=True,
            )
            labels = processor.tokenizer(list(example["text"]))["input_ids"]
            example["input_features"] = features["input_features"]
            if "attention_mask" in features:
                example["attention_mask"] = features["attention_mask"]
            example["labels"] = labels
            return example

        audio = load_audio_for_training(Path(str(audio_path)))
        features = processor.feature_extractor(
            audio,
            sampling_rate=TARGET_SAMPLING_RATE,
            return_attention_mask=True,
        )
        labels = processor.tokenizer(str(example["text"]))["input_ids"]
        example["input_features"] = features["input_features"][0]
        if "attention_mask" in features:
            example["attention_mask"] = features["attention_mask"][0]
        example["labels"] = labels
        return example

    return transform


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        model_input_name = self.processor.model_input_names[0]
        input_features = []
        for feature in features:
            padded_feature = {model_input_name: feature[model_input_name]}
            if "attention_mask" in feature:
                padded_feature["attention_mask"] = feature["attention_mask"]
            input_features.append(padded_feature)
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)

        if labels.shape[1] > 0 and torch.all(labels[:, 0] == self.decoder_start_token_id):
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def build_compute_metrics(processor: ProcessorLike):
    from evaluate import load

    wer_metric = load("wer")
    cer_metric = load("cer")

    def compute_metrics(prediction: EvalPredictionLike) -> dict[str, float]:
        prediction_ids = (
            prediction.predictions[0]
            if isinstance(prediction.predictions, tuple)
            else prediction.predictions
        )
        label_ids = prediction.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        predicted_texts = processor.batch_decode(prediction_ids, skip_special_tokens=True)
        reference_texts = processor.batch_decode(label_ids, skip_special_tokens=True)

        normalized_predictions = [normalize_transcript(text) for text in predicted_texts]
        normalized_references = [normalize_transcript(text) for text in reference_texts]

        return {
            "wer": wer_metric.compute(
                predictions=normalized_predictions,
                references=normalized_references,
            ),
            "cer": cer_metric.compute(
                predictions=normalized_predictions,
                references=normalized_references,
            ),
        }

    return compute_metrics


def create_runtime_datasets(
    train_rows: list[ManifestRow],
    valid_rows: list[ManifestRow],
    processor: ProcessorLike,
) -> tuple[Any, Any]:
    from datasets import Dataset

    train_dataset = Dataset.from_list([asdict(row) for row in train_rows])
    valid_dataset = Dataset.from_list([asdict(row) for row in valid_rows])

    transform = make_transform(processor)
    train_dataset.set_transform(transform)
    valid_dataset.set_transform(transform)
    return train_dataset, valid_dataset


def detect_environment(precision_preference: str) -> EnvironmentSnapshot:
    import datasets
    import torch
    import transformers

    has_cuda = torch.cuda.is_available()
    bf16_supported = has_cuda and torch.cuda.is_bf16_supported()
    precision_plan = resolve_precision_plan(precision_preference, has_cuda, bf16_supported)

    gpu_name = "cpu"
    gpu_total_memory_gb = 0.0
    device = "cpu"
    if has_cuda:
        current_device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(current_device)
        gpu_name = props.name
        gpu_total_memory_gb = props.total_memory / (1024**3)
        device = f"cuda:{current_device}"

    ram_gb = 0.0
    if (
        hasattr(os, "sysconf")
        and "SC_PAGE_SIZE" in os.sysconf_names
        and "SC_PHYS_PAGES" in os.sysconf_names
    ):
        ram_gb = (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / (1024**3)

    return EnvironmentSnapshot(
        device=device,
        gpu_name=gpu_name,
        gpu_total_memory_gb=round(gpu_total_memory_gb, 2),
        cpu_count=os.cpu_count() or 0,
        ram_gb=round(ram_gb, 2),
        precision=precision_plan.label,
        torch_version=torch.__version__,
        transformers_version=transformers.__version__,
        datasets_version=datasets.__version__,
    )


def set_global_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_decoder_start_token_id(model: Any) -> int:
    decoder_start_token_id = model.config.decoder_start_token_id
    if decoder_start_token_id is None:
        raise ValueError("Whisper model config is missing decoder_start_token_id.")
    return int(decoder_start_token_id)


def build_trainer(
    config: TrainingConfig,
    environment: EnvironmentSnapshot,
    train_rows: list[ManifestRow],
    valid_rows: list[ManifestRow],
):
    import torch
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    set_global_seed(config.seed)

    processor = WhisperProcessor.from_pretrained(
        config.model_name,
        language=config.language,
        task=config.task,
    )
    model = WhisperForConditionalGeneration.from_pretrained(config.model_name)
    model.generation_config.language = config.language
    model.generation_config.task = config.task
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None

    if config.gradient_checkpointing:
        model.config.use_cache = False

    precision_plan = resolve_precision_plan(
        config.precision,
        has_cuda=torch.cuda.is_available(),
        bf16_supported=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    )
    warmup_steps = compute_warmup_steps(
        train_rows=len(train_rows),
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        warmup_ratio=config.warmup_ratio,
    )

    train_dataset, valid_dataset = create_runtime_datasets(train_rows, valid_rows, processor)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=resolve_decoder_start_token_id(model),
    )
    compute_metrics = build_compute_metrics(processor)

    optim = "adamw_torch_fused" if environment.device.startswith("cuda") else "adamw_torch"
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        run_name=config.run_name,
        seed=config.seed,
        data_seed=config.seed,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        warmup_steps=warmup_steps,
        logging_strategy="steps",
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        predict_with_generate=True,
        generation_max_length=config.generation_max_length,
        metric_for_best_model="wer",
        greater_is_better=False,
        load_best_model_at_end=True,
        train_sampling_strategy="group_by_length" if config.group_by_length else "random",
        length_column_name="duration",
        remove_unused_columns=False,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=environment.device.startswith("cuda"),
        dataloader_persistent_workers=config.dataloader_num_workers > 0,
        gradient_checkpointing=config.gradient_checkpointing,
        bf16=precision_plan.bf16,
        fp16=precision_plan.fp16,
        report_to="none",
        optim=optim,
        tf32=environment.device.startswith("cuda"),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        processing_class=processor,
        compute_metrics=compute_metrics,
    )
    return trainer


def run_training(config: TrainingConfig) -> TrainingRunResult:
    environment = detect_environment(config.precision)

    train_source = Path(config.train_csv)
    valid_source = Path(config.valid_csv)
    reports_dir = Path(config.reports_dir)
    artifacts = create_artifacts(config.run_name, reports_dir)

    all_train_rows = load_manifest_rows(train_source)
    all_valid_rows = load_manifest_rows(valid_source)
    train_rows = select_rows(
        all_train_rows,
        sample_size=config.train_samples,
        seed=config.seed,
        max_duration_seconds=config.max_duration_seconds,
    )
    valid_rows = select_rows(
        all_valid_rows,
        sample_size=config.valid_samples,
        seed=config.seed + 1,
        max_duration_seconds=config.max_duration_seconds,
    )

    train_profile = build_dataset_profile(
        split_name="train",
        source_csv=train_source,
        selected_rows=train_rows,
        requested_rows=config.train_samples,
        total_available_rows=len(all_train_rows),
    )
    valid_profile = build_dataset_profile(
        split_name="valid",
        source_csv=valid_source,
        selected_rows=valid_rows,
        requested_rows=config.valid_samples,
        total_available_rows=len(all_valid_rows),
    )

    save_run_artifacts(
        artifacts=artifacts,
        config=config,
        environment=environment,
        train_profile=train_profile,
        valid_profile=valid_profile,
        train_rows=train_rows,
        valid_rows=valid_rows,
    )

    trainer = build_trainer(
        config=config,
        environment=environment,
        train_rows=train_rows,
        valid_rows=valid_rows,
    )
    train_result = trainer.train(
        resume_from_checkpoint=config.resume_from_checkpoint or None,
    )
    trainer.save_model()
    trainer.save_state()

    train_metrics = {
        key: float(value)
        for key, value in train_result.metrics.items()
        if isinstance(value, int | float)
    }
    eval_metrics = {
        key: float(value)
        for key, value in trainer.evaluate().items()
        if isinstance(value, int | float)
    }
    best_checkpoint = trainer.state.best_model_checkpoint or ""

    result = TrainingRunResult(
        config=config,
        environment=environment,
        train_profile=train_profile,
        valid_profile=valid_profile,
        train_metrics=train_metrics,
        eval_metrics=eval_metrics,
        best_checkpoint=best_checkpoint,
    )

    save_run_artifacts(
        artifacts=artifacts,
        config=config,
        environment=environment,
        train_profile=train_profile,
        valid_profile=valid_profile,
        train_rows=train_rows,
        valid_rows=valid_rows,
        train_metrics=train_metrics,
        eval_metrics=eval_metrics,
        summary_markdown=build_summary_markdown(result),
    )
    append_experiment_history(
        reports_dir,
        config,
        environment,
        train_profile,
        valid_profile,
        eval_metrics,
        best_checkpoint,
    )
    return result


def print_run_summary(result: TrainingRunResult) -> None:
    effective_batch_size = (
        result.config.per_device_train_batch_size * result.config.gradient_accumulation_steps
    )
    print(f"Run name: {result.config.run_name}")
    print(f"Output dir: {result.config.output_dir}")
    print(
        "Train subset:",
        f"{result.train_profile.selected_rows} rows",
        f"({result.train_profile.total_hours:.3f} h)",
    )
    print(
        "Valid subset:",
        f"{result.valid_profile.selected_rows} rows",
        f"({result.valid_profile.total_hours:.3f} h)",
    )
    print(f"Device: {result.environment.device}")
    print(f"Precision: {result.environment.precision}")
    print(f"Effective train batch size: {effective_batch_size}")
    print("Validation metrics:")
    print(f"  eval_wer: {format_metric(result.eval_metrics.get('eval_wer', float('nan')))}")
    print(f"  eval_cer: {format_metric(result.eval_metrics.get('eval_cer', float('nan')))}")
    print(f"Best checkpoint: {result.best_checkpoint or 'n/a'}")
    summary_path = create_artifacts(
        result.config.run_name,
        Path(result.config.reports_dir),
    ).summary_path
    print(f"Report summary: {summary_path}")


def main() -> None:
    config = resolve_training_config(parse_args())
    result = run_training(config)
    print_run_summary(result)


if __name__ == "__main__":
    main()
