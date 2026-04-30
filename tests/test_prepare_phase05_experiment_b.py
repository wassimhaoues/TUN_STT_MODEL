from dataset.prepare_phase05_experiment_b import (
    AudioStats,
    ManifestRow,
    build_training_audit_candidates,
    percentile,
    suggested_audio_action,
)


def make_row(sample_id: str, text: str, duration_seconds: float) -> ManifestRow:
    return ManifestRow(
        split="train",
        sample_id=sample_id,
        text=text,
        duration_seconds=duration_seconds,
        wav_path=f"dataset/extracted_wavs/{sample_id}.wav",
    )


def test_percentile_returns_expected_ranked_value() -> None:
    values = [-40.0, -35.0, -30.0, -25.0, -20.0]
    assert percentile(values, 0.10) == -40.0
    assert percentile(values, 0.50) == -30.0


def test_suggested_audio_action_requires_headroom_for_gain() -> None:
    assert (
        suggested_audio_action(AudioStats(rms_dbfs=-34.0, peak_dbfs=-10.0), -30.0)
        == "consider_gain_normalize"
    )
    assert (
        suggested_audio_action(AudioStats(rms_dbfs=-34.0, peak_dbfs=-2.0), -30.0)
        == "review_before_gain_change"
    )
    assert suggested_audio_action(AudioStats(rms_dbfs=-25.0, peak_dbfs=-4.0), -30.0) == "keep_raw"


def test_build_training_audit_candidates_focuses_on_low_volume_target_rows() -> None:
    rows = [
        make_row("sample_1", "bonjour", 2.5),
        make_row("sample_2", "مرحبا", 6.0),
        make_row("sample_3", "اهلا", 1.5),
        make_row("sample_4", "اهلا", 6.0),
    ]
    audio_stats_by_id = {
        "sample_1": AudioStats(rms_dbfs=-36.0, peak_dbfs=-12.0),
        "sample_2": AudioStats(rms_dbfs=-38.0, peak_dbfs=-11.0),
        "sample_3": AudioStats(rms_dbfs=-31.0, peak_dbfs=-9.0),
        "sample_4": AudioStats(rms_dbfs=-24.0, peak_dbfs=-3.0),
    }

    candidates = build_training_audit_candidates(
        rows=rows,
        audio_stats_by_id=audio_stats_by_id,
        short_clip_threshold=3.0,
        low_volume_dbfs=-32.0,
        very_low_volume_dbfs=-37.0,
        limit=10,
    )

    candidate_ids = [candidate.row.sample_id for candidate in candidates]
    assert candidate_ids == ["sample_2", "sample_1"]
    assert candidates[0].selection_reason == "very_low_volume"
    assert candidates[1].selection_reason == "low_volume|code_switched|short_clip"
