from pathlib import Path

from training.analyze_errors import (
    analyze_prediction_row,
    build_analysis_report,
    build_bucket_summary,
    build_cer_review_counts,
    build_summary_markdown,
    has_repetition_loop,
    max_repetition_run,
    resolve_cer_review_band,
    resolve_critical_review_reason,
    resolve_output_dir,
    select_cer_review_queue,
)


def test_has_repetition_loop_detects_repeated_tokens() -> None:
    assert has_repetition_loop(["انا", "انا"])
    assert has_repetition_loop(["وقتها", "وقت", "وقتها", "وقت"])
    assert not has_repetition_loop(["انا", "نمشي", "غدوة"])
    assert max_repetition_run(["مال", "مال", "غيره"]) == 2
    assert max_repetition_run(["وحلت", "معاها", "وحلت", "معاها", "وحلت", "معاها"]) == 3


def test_analyze_prediction_row_flags_major_issues() -> None:
    prediction_row = {
        "id": "sample_1",
        "wav_path": "dataset/extracted_wavs/sample_1.wav",
        "reference": "genre تنجموا كان تحبوا a part ca",
        "prediction": "تنجموا تنجموا تنجموا",
    }
    source_row = {"id": "sample_1", "duration": "2.100"}

    sample = analyze_prediction_row(prediction_row, source_row)

    assert "code_switched_reference" in sample.bucket_flags
    assert "short_clip" in sample.bucket_flags
    assert "major_omission" in sample.bucket_flags
    assert "repeated_token_hallucination" in sample.bucket_flags
    assert sample.cer_review_band == "critical"
    assert sample.critical_review_reason == "repeated_words"


def test_repetition_hallucination_tolerates_small_reference_repetition() -> None:
    prediction_row = {
        "id": "sample_2",
        "wav_path": "dataset/extracted_wavs/sample_2.wav",
        "reference": "مال مال غيره",
        "prediction": "انا انا انا انا انا",
    }
    source_row = {"id": "sample_2", "duration": "5.000"}

    sample = analyze_prediction_row(prediction_row, source_row)

    assert "repeated_token_hallucination" in sample.bucket_flags
    assert "catastrophic_looping" not in sample.bucket_flags


def test_build_bucket_summary_returns_nan_for_empty_bucket() -> None:
    summary = build_bucket_summary([], "short_clip")

    assert summary.sample_count == 0
    assert summary.sample_rate == 0.0
    assert summary.description


def test_build_analysis_report_and_markdown() -> None:
    prediction_row = {
        "id": "sample_1",
        "wav_path": "dataset/extracted_wavs/sample_1.wav",
        "reference": "c'est bon",
        "prediction": "so",
    }
    source_row = {"id": "sample_1", "duration": "0.670"}
    sample = analyze_prediction_row(prediction_row, source_row)

    report = build_analysis_report(
        Path("reports/runs/demo-run/predictions.csv"),
        Path("dataset/metadata_test.csv"),
        [sample],
    )
    markdown = build_summary_markdown(report)

    assert report.total_samples == 1
    assert report.manual_review_candidates
    assert "Phase 04 Error Analysis" in markdown
    assert "Main Findings" in markdown
    assert "CER Review Bands" in markdown


def test_resolve_output_dir_for_run_predictions() -> None:
    predictions_csv = Path("/repo/reports/runs/demo-run/predictions.csv")
    output_dir = resolve_output_dir(predictions_csv)

    assert output_dir == predictions_csv.parent / "error_analysis"


def test_cer_review_band_thresholds_and_critical_override() -> None:
    assert resolve_cer_review_band(0.10, "") == "good"
    assert resolve_cer_review_band(0.20, "") == "acceptable"
    assert resolve_cer_review_band(0.30, "") == "needs_review"
    assert resolve_cer_review_band(0.45, "") == "high_priority_review"
    assert resolve_cer_review_band(0.05, "empty_prediction") == "critical"


def test_critical_review_reason_detects_wrong_language() -> None:
    reason = resolve_critical_review_reason(
        prediction="bonjour",
        repeated_hallucination=False,
        catastrophic_looping=False,
        major_omission=False,
        wrong_language_prediction=True,
    )
    assert reason == "wrong_language"


def test_cer_review_queue_prioritizes_critical_then_high_cer() -> None:
    source_row = {"id": "sample_x", "duration": "5.000"}
    samples = [
        analyze_prediction_row(
            {
                "id": "sample_1",
                "wav_path": "dataset/extracted_wavs/sample_1.wav",
                "reference": "مرحبا بيك",
                "prediction": "bonjour",
            },
            source_row,
        ),
        analyze_prediction_row(
            {
                "id": "sample_2",
                "wav_path": "dataset/extracted_wavs/sample_2.wav",
                "reference": "مرحبا بيك",
                "prediction": "مرحبا فيك جدا",
            },
            source_row,
        ),
    ]

    queue = select_cer_review_queue(samples)
    counts = build_cer_review_counts(samples)

    assert queue[0].sample_id == "sample_1"
    assert counts["critical"] == 1
