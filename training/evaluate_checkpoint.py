from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataset.text_normalization import NORMALIZATION_VERSION, normalize_transcript  # noqa: E402
from training.baseline_test import (  # noqa: E402
    HISTORY_COLUMNS,
    PredictionRecord,
    format_metric,
    get_device_config,
    load_audio_for_asr,
    write_predictions_csv,
)

if TYPE_CHECKING:
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

DATASET_DIR = ROOT_DIR / "dataset"
WAV_DIR = DATASET_DIR / "extracted_wavs"
DEFAULT_SOURCE_CSV = DATASET_DIR / "metadata_test.csv"
REPORTS_DIR = ROOT_DIR / "reports"
HISTORY_CSV = REPORTS_DIR / "experiment_history.csv"
DEFAULT_LANGUAGE = "arabic"
DEFAULT_TASK = "transcribe"
DEFAULT_RUN_TYPE = "checkpoint_eval"
DEFAULT_QUICK_SAMPLES = 20


@dataclass(frozen=True)
class EvaluationRunResult:
    run_name: str
    run_type: str
    model_name: str
    eval_scope: str
    n_samples: int
    wer: float
    cer: float
    device: str
    language: str
    task: str
    source_csv: str
    created_at: str
    notes: str
    predictions: list[PredictionRecord]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a local Whisper checkpoint and save tracked reports."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Checkpoint directory or model path to evaluate.",
    )
    parser.add_argument(
        "--run-name",
        help="Optional run name. Defaults to a timestamped checkpoint-eval name.",
    )
    parser.add_argument(
        "--run-type",
        default=DEFAULT_RUN_TYPE,
        help=f"Tracked run type label. Defaults to {DEFAULT_RUN_TYPE}.",
    )
    parser.add_argument(
        "--source-csv",
        default=str(DEFAULT_SOURCE_CSV),
        help="CSV manifest to evaluate. Defaults to dataset/metadata_test.csv.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help="Rows to evaluate from the head of the CSV. Use 0 for the full file.",
    )
    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help="Whisper generation language.",
    )
    parser.add_argument(
        "--task",
        default=DEFAULT_TASK,
        help="Whisper generation task.",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Optional notes to save in experiment history and summary.",
    )
    return parser.parse_args()


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return cleaned or "run"


def build_run_name(model_path: str, created_at: datetime, samples: int) -> str:
    model_slug = sanitize_name(Path(model_path).name or model_path)
    scope = "test-full" if samples == 0 else f"test-head-{samples}"
    timestamp = created_at.strftime("%Y%m%d-%H%M%S")
    return f"{model_slug}-{scope}-{timestamp}"


def get_eval_scope(csv_path: Path | str, sample_count: int, total_rows: int) -> str:
    csv_path = Path(csv_path)
    if csv_path.resolve() == DEFAULT_SOURCE_CSV.resolve():
        return "test_full" if sample_count == total_rows else f"test_head_{sample_count}"
    if sample_count == total_rows:
        return f"{csv_path.stem}_full"
    return f"{csv_path.stem}_head_{sample_count}"


def build_history_row(result: EvaluationRunResult) -> dict[str, str]:
    return {
        "run_name": result.run_name,
        "run_type": result.run_type,
        "model_name": result.model_name,
        "eval_scope": result.eval_scope,
        "n_samples": str(result.n_samples),
        "wer": format_metric(result.wer),
        "cer": format_metric(result.cer),
        "device": result.device,
        "language": result.language,
        "task": result.task,
        "source_csv": result.source_csv,
        "created_at": result.created_at,
        "notes": result.notes,
    }


def append_history_row(history_path: Path, result: EvaluationRunResult) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not history_path.exists()
    with history_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=HISTORY_COLUMNS,
            lineterminator="\n",
        )
        if write_header:
            writer.writeheader()
        writer.writerow(build_history_row(result))


def build_summary_markdown(result: EvaluationRunResult) -> str:
    note_text = result.notes or "No additional notes."
    return "\n".join(
        [
            f"# Checkpoint Evaluation Report: {result.run_name}",
            "",
            "This report records a fine-tuned checkpoint evaluation against a tracked manifest.",
            "Use it to compare quick and locked test performance against the raw baseline.",
            "",
            "## Run Details",
            "",
            f"- Run name: `{result.run_name}`",
            f"- Run type: `{result.run_type}`",
            f"- Model path: `{result.model_name}`",
            f"- Date: `{result.created_at}`",
            f"- Dataset source: `{result.source_csv}`",
            f"- Evaluation scope: `{result.eval_scope}`",
            f"- Sample count: `{result.n_samples}`",
            f"- Device: `{result.device}`",
            f"- Language: `{result.language}`",
            f"- Task: `{result.task}`",
            f"- Metric normalization: `{NORMALIZATION_VERSION}`",
            "",
            "## Metrics",
            "",
            f"- WER: `{format_metric(result.wer)}`",
            f"- CER: `{format_metric(result.cer)}`",
            "",
            "## Notes",
            "",
            note_text,
            "",
        ]
    )


def save_run_report(
    result: EvaluationRunResult,
    reports_dir: Path = REPORTS_DIR,
) -> tuple[Path, Path]:
    history_path = reports_dir / "experiment_history.csv"
    run_dir = reports_dir / "runs" / result.run_name
    summary_path = run_dir / "summary.md"
    predictions_path = run_dir / "predictions.csv"

    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(build_summary_markdown(result), encoding="utf-8")
    write_predictions_csv(predictions_path, result.predictions)
    append_history_row(history_path, result)

    return summary_path, predictions_path


def transcribe_audio(
    processor: WhisperProcessor,
    model: WhisperForConditionalGeneration,
    audio_input: dict[str, object],
    model_device: str,
    language: str,
    task: str,
) -> str:
    import torch

    inputs = processor(
        audio_input["raw"],
        sampling_rate=audio_input["sampling_rate"],
        return_tensors="pt",
    )
    input_features = inputs.input_features.to(model_device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_features,
            language=language,
            task=task,
        )

    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def run_evaluation(
    model_path: str,
    source_csv: Path,
    samples: int,
    run_name: str | None,
    run_type: str,
    language: str,
    task: str,
    notes: str,
) -> EvaluationRunResult:
    import pandas as pd
    from evaluate import load
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    if samples < 0:
        raise ValueError("--samples must be 0 or greater.")
    if not source_csv.exists():
        raise FileNotFoundError(f"Missing evaluation CSV: {source_csv}")

    created_at = datetime.now().astimezone()
    resolved_run_name = run_name or build_run_name(model_path, created_at, samples)
    device_id, device_label = get_device_config()

    full_df = pd.read_csv(source_csv)
    if full_df.empty:
        raise ValueError(f"No rows found in {source_csv}.")
    total_rows = len(full_df)
    df = full_df
    if samples > 0:
        df = df.head(samples)

    processor = WhisperProcessor.from_pretrained(
        model_path,
        language=language,
        task=task,
    )
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model.generation_config.language = language
    model.generation_config.task = task
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model_device = "cuda" if device_id >= 0 else "cpu"
    model = model.to(model_device)
    model.eval()

    wer_metric = load("wer")
    cer_metric = load("cer")
    prediction_rows: list[PredictionRecord] = []
    references: list[str] = []
    predictions: list[str] = []

    for row in df.itertuples(index=False):
        wav_path = WAV_DIR / f"{row.id}.wav"
        if not wav_path.exists():
            raise FileNotFoundError(f"Missing WAV file: {wav_path}")

        reference = str(row.text)
        prediction = transcribe_audio(
            processor=processor,
            model=model,
            audio_input=load_audio_for_asr(wav_path),
            model_device=model_device,
            language=language,
            task=task,
        )
        predictions.append(normalize_transcript(prediction))
        references.append(normalize_transcript(reference))
        prediction_rows.append(
            PredictionRecord(
                id=str(row.id),
                wav_path=str(wav_path.relative_to(ROOT_DIR)),
                reference=reference,
                prediction=prediction,
            )
        )

        print("\nID:", row.id)
        print("REF:", reference)
        print("PRED:", prediction)
        print("-" * 80)

    wer = float(wer_metric.compute(predictions=predictions, references=references))
    cer = float(cer_metric.compute(predictions=predictions, references=references))

    return EvaluationRunResult(
        run_name=resolved_run_name,
        run_type=run_type,
        model_name=model_path,
        eval_scope=get_eval_scope(source_csv, len(prediction_rows), total_rows),
        n_samples=len(prediction_rows),
        wer=wer,
        cer=cer,
        device=device_label,
        language=language,
        task=task,
        source_csv=str(source_csv),
        created_at=created_at.isoformat(),
        notes=notes,
        predictions=prediction_rows,
    )


def main() -> None:
    args = parse_args()
    result = run_evaluation(
        model_path=args.model_path,
        source_csv=Path(args.source_csv),
        samples=args.samples,
        run_name=args.run_name,
        run_type=args.run_type,
        language=args.language,
        task=args.task,
        notes=args.notes,
    )
    summary_path, predictions_path = save_run_report(result)

    print("\n======================")
    print("CHECKPOINT EVALUATION RESULT")
    print("======================")
    print("Run name:", result.run_name)
    print("Samples:", result.n_samples)
    print("WER:", result.wer)
    print("CER:", result.cer)
    print("Summary saved to:", summary_path)
    print("Predictions saved to:", predictions_path)
    print("History updated at:", HISTORY_CSV)


if __name__ == "__main__":
    main()
