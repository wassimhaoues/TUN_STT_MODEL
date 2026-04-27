import argparse
import csv
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
from evaluate import load
from scipy.signal import resample_poly
from transformers import WhisperForConditionalGeneration, WhisperProcessor

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataset.text_normalization import NORMALIZATION_VERSION, normalize_transcript  # noqa: E402

DATASET_DIR = ROOT_DIR / "dataset"
WAV_DIR = DATASET_DIR / "extracted_wavs"
TEST_CSV = DATASET_DIR / "metadata_test.csv"
REPORTS_DIR = ROOT_DIR / "reports"
RUNS_DIR = REPORTS_DIR / "runs"
HISTORY_CSV = REPORTS_DIR / "experiment_history.csv"

MODEL_NAME = "openai/whisper-small"
DEFAULT_RUN_TYPE = "baseline"
DEFAULT_LANGUAGE = "arabic"
DEFAULT_TASK = "transcribe"
DEFAULT_SAMPLES = 20
TARGET_SAMPLING_RATE = 16000
SOURCE_CSV = Path("dataset/metadata_test.csv")
HISTORY_COLUMNS = [
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
class PredictionRecord:
    id: str
    wav_path: str
    reference: str
    prediction: str


@dataclass(frozen=True)
class BaselineRunResult:
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
        description="Run a Whisper Small baseline evaluation and save a tracked report."
    )
    parser.add_argument(
        "--run-name",
        help="Optional name for this run. Defaults to a timestamped Whisper Small baseline name.",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Optional notes to save in the experiment history and Markdown summary.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help=f"Number of test rows to evaluate. Defaults to {DEFAULT_SAMPLES}.",
    )
    return parser.parse_args()


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return cleaned or "run"


def build_run_name(model_name: str, created_at: datetime) -> str:
    model_slug = sanitize_name(model_name.split("/")[-1])
    timestamp = created_at.strftime("%Y%m%d-%H%M%S")
    return f"{model_slug}-baseline-{timestamp}"


def get_device_config() -> tuple[int, str]:
    if torch.cuda.is_available():
        return 0, f"cuda:{torch.cuda.current_device()}"
    return -1, "cpu"


def get_eval_scope(samples: int) -> str:
    return f"test_head_{samples}"


def format_metric(value: float) -> str:
    return f"{value:.6f}"


def load_audio_for_asr(wav_path: Path) -> dict[str, object]:
    audio, sample_rate = sf.read(str(wav_path))
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    if sample_rate != TARGET_SAMPLING_RATE:
        audio = resample_poly(audio, TARGET_SAMPLING_RATE, sample_rate)
        sample_rate = TARGET_SAMPLING_RATE
    return {
        "raw": audio.astype("float32", copy=False),
        "sampling_rate": sample_rate,
    }


def transcribe_audio(
    processor: WhisperProcessor,
    model: WhisperForConditionalGeneration,
    audio_input: dict[str, object],
    model_device: str,
) -> str:
    inputs = processor(
        audio_input["raw"],
        sampling_rate=audio_input["sampling_rate"],
        return_tensors="pt",
    )
    input_features = inputs.input_features.to(model_device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_features,
            language=DEFAULT_LANGUAGE,
            task=DEFAULT_TASK,
        )

    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def build_history_row(result: BaselineRunResult) -> dict[str, str]:
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


def build_summary_markdown(result: BaselineRunResult) -> str:
    note_text = result.notes or "No additional notes."
    return "\n".join(
        [
            f"# Baseline Report: {result.run_name}",
            "",
            "This report records the starting Whisper Small baseline before fine-tuning.",
            "Use it as the reference point for future step-by-step training improvements.",
            "",
            "## Run Details",
            "",
            f"- Run name: `{result.run_name}`",
            f"- Run type: `{result.run_type}`",
            f"- Model: `{result.model_name}`",
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


def append_history_row(history_path: Path, result: BaselineRunResult) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not history_path.exists()

    with history_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(build_history_row(result))


def write_predictions_csv(predictions_path: Path, predictions: list[PredictionRecord]) -> None:
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "wav_path", "reference", "prediction"],
        )
        writer.writeheader()
        for record in predictions:
            writer.writerow(
                {
                    "id": record.id,
                    "wav_path": record.wav_path,
                    "reference": record.reference,
                    "prediction": record.prediction,
                }
            )


def save_run_report(
    result: BaselineRunResult, reports_dir: Path = REPORTS_DIR
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


def run_baseline(samples: int, run_name: str | None = None, notes: str = "") -> BaselineRunResult:
    if samples <= 0:
        raise ValueError("--samples must be greater than 0.")

    created_at = datetime.now().astimezone()
    resolved_run_name = run_name or build_run_name(MODEL_NAME, created_at)
    device_id, device_label = get_device_config()
    df = pd.read_csv(TEST_CSV).head(samples)

    if df.empty:
        raise ValueError(f"No rows found in {TEST_CSV} for baseline evaluation.")

    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME,
        language=DEFAULT_LANGUAGE,
        task=DEFAULT_TASK,
    )
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
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

    return BaselineRunResult(
        run_name=resolved_run_name,
        run_type=DEFAULT_RUN_TYPE,
        model_name=MODEL_NAME,
        eval_scope=get_eval_scope(len(prediction_rows)),
        n_samples=len(prediction_rows),
        wer=wer,
        cer=cer,
        device=device_label,
        language=DEFAULT_LANGUAGE,
        task=DEFAULT_TASK,
        source_csv=str(SOURCE_CSV),
        created_at=created_at.isoformat(),
        notes=notes,
        predictions=prediction_rows,
    )


def main() -> None:
    args = parse_args()
    result = run_baseline(samples=args.samples, run_name=args.run_name, notes=args.notes)
    summary_path, predictions_path = save_run_report(result)

    print("\n======================")
    print("BASELINE RESULT")
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
