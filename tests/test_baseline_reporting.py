from __future__ import annotations

import csv

from training.baseline_test import (
    BaselineRunResult,
    PredictionRecord,
    build_summary_markdown,
    save_run_report,
)


def make_result(run_name: str, notes: str, wer: float) -> BaselineRunResult:
    return BaselineRunResult(
        run_name=run_name,
        run_type="baseline",
        model_name="openai/whisper-small",
        eval_scope="test_head_20",
        n_samples=2,
        wer=wer,
        cer=0.222222,
        device="cpu",
        language="arabic",
        task="transcribe",
        source_csv="dataset/metadata_test.csv",
        created_at="2026-04-27T20:45:00+01:00",
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


def test_build_summary_markdown_includes_key_run_details() -> None:
    result = make_result("baseline-start", "Raw Whisper Small before fine-tuning.", 0.111111)

    summary = build_summary_markdown(result)

    assert "# Baseline Report: baseline-start" in summary
    assert "starting Whisper Small baseline before fine-tuning" in summary
    assert "`openai/whisper-small`" in summary
    assert "`2`" in summary
    assert "`0.111111`" in summary
    assert "`0.222222`" in summary
    assert "Raw Whisper Small before fine-tuning." in summary


def test_save_run_report_writes_summary_predictions_and_appends_history(tmp_path) -> None:
    first = make_result("baseline-one", "first run", 0.100000)
    second = make_result("baseline-two", "second run", 0.200000)

    summary_path, predictions_path = save_run_report(first, reports_dir=tmp_path)
    save_run_report(second, reports_dir=tmp_path)

    assert summary_path.exists()
    assert predictions_path.exists()

    with predictions_path.open("r", encoding="utf-8", newline="") as handle:
        prediction_rows = list(csv.DictReader(handle))

    assert len(prediction_rows) == 2
    assert prediction_rows[0]["id"] == "sample_1"
    assert prediction_rows[0]["wav_path"] == "dataset/extracted_wavs/sample_1.wav"
    assert prediction_rows[0]["reference"] == "ahla"
    assert prediction_rows[0]["prediction"] == "ahlan"

    history_path = tmp_path / "experiment_history.csv"
    with history_path.open("r", encoding="utf-8", newline="") as handle:
        history_rows = list(csv.DictReader(handle))

    assert len(history_rows) == 2
    assert history_rows[0]["run_name"] == "baseline-one"
    assert history_rows[1]["run_name"] == "baseline-two"
    assert history_rows[0]["notes"] == "first run"
    assert history_rows[1]["wer"] == "0.200000"
