from __future__ import annotations

import csv

from training.baseline_test import PredictionRecord
from training.evaluate_checkpoint import (
    EvaluationRunResult,
    build_summary_markdown,
    get_eval_scope,
    save_run_report,
)


def make_result(run_name: str, notes: str, wer: float) -> EvaluationRunResult:
    return EvaluationRunResult(
        run_name=run_name,
        run_type="phase03_full_eval",
        model_name="outputs/train_runs/whisper-small-phase03-full",
        eval_scope="test_full",
        n_samples=4,
        wer=wer,
        cer=0.222222,
        device="cuda:0",
        language="arabic",
        task="transcribe",
        source_csv="dataset/metadata_test.csv",
        created_at="2026-04-28T00:10:00+01:00",
        notes=notes,
        predictions=[
            PredictionRecord(
                id="sample_1",
                wav_path="dataset/extracted_wavs/sample_1.wav",
                reference="ahla",
                prediction="ahlan",
            ),
            PredictionRecord(
                id="sample_2",
                wav_path="dataset/extracted_wavs/sample_2.wav",
                reference="labes",
                prediction="lebes",
            ),
        ],
    )


def test_build_summary_markdown_includes_checkpoint_details() -> None:
    result = make_result("phase03-full-test", "Locked test evaluation.", 0.111111)

    summary = build_summary_markdown(result)

    assert "# Checkpoint Evaluation Report: phase03-full-test" in summary
    assert "fine-tuned checkpoint evaluation" in summary
    assert "`outputs/train_runs/whisper-small-phase03-full`" in summary
    assert "`0.111111`" in summary
    assert "Locked test evaluation." in summary


def test_get_eval_scope_distinguishes_head_and_full() -> None:
    assert get_eval_scope("dataset/metadata_test.csv", 20, 1036) == "test_head_20"
    assert get_eval_scope("dataset/metadata_test.csv", 1036, 1036) == "test_full"
    assert get_eval_scope("dataset/metadata_valid.csv", 100, 100) == "metadata_valid_full"


def test_save_run_report_writes_summary_predictions_and_history(tmp_path) -> None:
    result = make_result("phase03-quick-test", "Quick benchmark.", 0.333333)

    summary_path, predictions_path = save_run_report(result, reports_dir=tmp_path)

    assert summary_path.exists()
    assert predictions_path.exists()

    with predictions_path.open("r", encoding="utf-8", newline="") as handle:
        prediction_rows = list(csv.DictReader(handle))

    assert len(prediction_rows) == 2
    assert prediction_rows[0]["prediction"] == "ahlan"

    history_path = tmp_path / "experiment_history.csv"
    with history_path.open("r", encoding="utf-8", newline="") as handle:
        history_rows = list(csv.DictReader(handle))

    assert len(history_rows) == 1
    assert history_rows[0]["run_name"] == "phase03-quick-test"
    assert history_rows[0]["wer"] == "0.333333"
    assert history_rows[0]["run_type"] == "phase03_full_eval"
