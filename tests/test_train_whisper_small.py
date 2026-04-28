from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import pytest

from training.train_whisper_small import (
    DatasetProfile,
    EnvironmentSnapshot,
    ManifestRow,
    TrainingConfig,
    TrainingRunResult,
    append_experiment_history,
    build_summary_markdown,
    create_artifacts,
    resolve_precision_plan,
    save_run_artifacts,
    select_rows,
    validate_training_config,
)


def make_row(tmp_path: Path, sample_id: str, duration: float, text: str = "ahla") -> ManifestRow:
    audio_path = tmp_path / f"{sample_id}.wav"
    audio_path.write_bytes(b"wav")
    return ManifestRow(
        id=sample_id,
        text=text,
        duration=duration,
        text_raw=text,
        normalization_changed=False,
        normalization_version="v1",
        audio_path=str(audio_path),
    )


def make_config(tmp_path: Path, run_name: str = "phase02-smoke") -> TrainingConfig:
    return TrainingConfig(
        run_name=run_name,
        run_type="phase02_smoke_train",
        model_name="openai/whisper-small",
        train_csv="dataset/metadata_train.csv",
        valid_csv="dataset/metadata_valid.csv",
        train_samples=3,
        valid_samples=2,
        seed=42,
        output_dir=str(tmp_path / "outputs"),
        reports_dir=str(tmp_path / "reports"),
        language="arabic",
        task="transcribe",
        precision="auto",
        gradient_checkpointing=True,
        group_by_length=True,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        num_train_epochs=1.0,
        max_steps=10,
        warmup_ratio=0.1,
        eval_steps=5,
        save_steps=5,
        logging_steps=1,
        save_total_limit=2,
        dataloader_num_workers=2,
        generation_max_length=225,
        max_duration_seconds=30.0,
        notes="phase 02 smoke",
        resume_from_checkpoint="",
        created_at="2026-04-27T22:00:00+01:00",
        git_commit="abc123",
        git_dirty=False,
    )


def make_environment() -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        device="cuda:0",
        gpu_name="RTX Test",
        gpu_total_memory_gb=8.0,
        cpu_count=24,
        ram_gb=16.0,
        precision="bf16",
        torch_version="2.10.0",
        transformers_version="5.6.2",
        datasets_version="4.8.5",
    )


def make_profile(split_name: str, source_csv: str, selected_rows: int) -> DatasetProfile:
    return DatasetProfile(
        split_name=split_name,
        source_csv=source_csv,
        selected_rows=selected_rows,
        requested_rows=selected_rows,
        total_available_rows=selected_rows * 10,
        total_hours=1.234,
        min_duration=1.111,
        median_duration=2.222,
        max_duration=3.333,
        normalization_version="v1",
        sample_ids=["sample_1", "sample_2"],
    )


def test_select_rows_is_reproducible_and_applies_duration_filter(tmp_path: Path) -> None:
    rows = [
        make_row(tmp_path, "sample_1", 4.0),
        make_row(tmp_path, "sample_2", 5.0),
        make_row(tmp_path, "sample_3", 40.0),
        make_row(tmp_path, "sample_4", 3.0),
    ]

    first = select_rows(rows, sample_size=2, seed=7, max_duration_seconds=10.0)
    second = select_rows(rows, sample_size=2, seed=7, max_duration_seconds=10.0)

    assert [row.id for row in first] == [row.id for row in second]
    assert all(row.duration <= 10.0 for row in first)
    assert "sample_3" not in {row.id for row in first}


def test_select_rows_zero_means_use_all_eligible_rows(tmp_path: Path) -> None:
    rows = [
        make_row(tmp_path, "sample_1", 4.0),
        make_row(tmp_path, "sample_2", 5.0),
        make_row(tmp_path, "sample_3", 40.0),
    ]

    selected = select_rows(rows, sample_size=0, seed=7, max_duration_seconds=10.0)

    assert [row.id for row in selected] == ["sample_1", "sample_2"]


def test_validate_training_config_rejects_zero_max_steps(tmp_path: Path) -> None:
    invalid = replace(make_config(tmp_path), max_steps=0)

    with pytest.raises(ValueError, match="max_steps cannot be 0"):
        validate_training_config(invalid)


def test_phase02_artifacts_and_history_are_written(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    environment = make_environment()
    train_rows = [
        make_row(tmp_path, "sample_1", 1.5),
        make_row(tmp_path, "sample_2", 2.5),
        make_row(tmp_path, "sample_3", 3.5),
    ]
    valid_rows = [
        make_row(tmp_path, "sample_4", 1.2),
        make_row(tmp_path, "sample_5", 2.2),
    ]
    train_profile = make_profile("train", config.train_csv, 3)
    valid_profile = make_profile("valid", config.valid_csv, 2)
    result = TrainingRunResult(
        config=config,
        environment=environment,
        train_profile=train_profile,
        valid_profile=valid_profile,
        train_metrics={"train_loss": 1.234567, "train_runtime": 12.0},
        eval_metrics={"eval_wer": 0.456789, "eval_cer": 0.234567},
        best_checkpoint=str(Path(config.output_dir) / "checkpoint-10"),
    )
    artifacts = create_artifacts(config.run_name, Path(config.reports_dir))

    save_run_artifacts(
        artifacts=artifacts,
        config=config,
        environment=environment,
        train_profile=train_profile,
        valid_profile=valid_profile,
        train_rows=train_rows,
        valid_rows=valid_rows,
        train_metrics=result.train_metrics,
        eval_metrics=result.eval_metrics,
        summary_markdown=build_summary_markdown(result),
    )
    append_experiment_history(
        reports_dir=Path(config.reports_dir),
        config=config,
        environment=environment,
        train_profile=train_profile,
        valid_profile=valid_profile,
        eval_metrics=result.eval_metrics,
        best_checkpoint=result.best_checkpoint,
    )

    assert artifacts.summary_path.exists()
    assert artifacts.config_path.exists()
    assert artifacts.environment_path.exists()
    assert artifacts.train_manifest_path.exists()
    assert artifacts.valid_manifest_path.exists()
    assert artifacts.dataset_profile_path.exists()
    assert artifacts.train_metrics_path.exists()
    assert artifacts.eval_metrics_path.exists()

    summary = artifacts.summary_path.read_text(encoding="utf-8")
    assert "Phase 02 end-to-end Whisper Small smoke fine-tuning run" in summary
    assert "`openai/whisper-small`" in summary
    assert "`0.456789`" in summary
    assert "checkpoint-10" in summary

    with (Path(config.reports_dir) / "experiment_history.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        history_rows = list(csv.DictReader(handle))

    assert len(history_rows) == 1
    assert history_rows[0]["run_name"] == config.run_name
    assert history_rows[0]["run_type"] == "phase02_smoke_train"
    assert history_rows[0]["wer"] == "0.456789"
    assert history_rows[0]["n_samples"] == "3"
    assert "train_samples=3" in history_rows[0]["notes"]


def test_resolve_precision_plan_prefers_bf16_then_fp16_then_fp32() -> None:
    assert resolve_precision_plan("auto", has_cuda=True, bf16_supported=True).label == "bf16"
    assert resolve_precision_plan("auto", has_cuda=True, bf16_supported=False).label == "fp16"
    assert resolve_precision_plan("auto", has_cuda=False, bf16_supported=False).label == "fp32"
