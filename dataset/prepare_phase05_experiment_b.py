from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

try:
    from dataset.build_phase05_manifests import DEFAULT_SHORT_CLIP_THRESHOLD, has_latin
except ModuleNotFoundError:  # pragma: no cover - script entrypoint fallback
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from dataset.build_phase05_manifests import DEFAULT_SHORT_CLIP_THRESHOLD, has_latin

ROOT_DIR = Path(__file__).resolve().parent.parent

DEFAULT_TRAIN_CSV = ROOT_DIR / "dataset" / "metadata_train.csv"
DEFAULT_VALID_CSV = ROOT_DIR / "dataset" / "metadata_valid.csv"
DEFAULT_TEST_CSV = ROOT_DIR / "dataset" / "metadata_test.csv"
DEFAULT_PREDICTIONS_CSV = (
    ROOT_DIR
    / "reports"
    / "runs"
    / "whisper-small-phase03-full-20260428-001840-phase05-safe-decode-locked"
    / "predictions.csv"
)
DEFAULT_ERROR_ANALYSIS_CSV = (
    ROOT_DIR
    / "reports"
    / "runs"
    / "whisper-small-phase03-full-20260428-001840-phase05-safe-decode-locked"
    / "error_analysis"
    / "detailed_rows.csv"
)
DEFAULT_MANUAL_REVIEW_CSV = (
    ROOT_DIR
    / "reports"
    / "runs"
    / "whisper-small-phase03-full-20260428-001840-phase05-safe-decode-locked"
    / "error_analysis"
    / "manual_review_candidates.csv"
)
DEFAULT_HISTORY_CSV = ROOT_DIR / "reports" / "experiment_history.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "reports" / "phase05_preparation"
DEFAULT_AUDIT_DIR = ROOT_DIR / "dataset" / "audits"
DEFAULT_TRAINING_AUDIT_LIMIT = 180
DEFAULT_BENCHMARK_AUDIT_LIMIT = 60


@dataclass(frozen=True)
class ManifestRow:
    split: str
    sample_id: str
    text: str
    duration_seconds: float
    wav_path: str


@dataclass(frozen=True)
class AudioStats:
    rms_dbfs: float
    peak_dbfs: float


@dataclass(frozen=True)
class TrainingAuditCandidate:
    row: ManifestRow
    audio_stats: AudioStats
    has_latin_text: bool
    is_short_clip: bool
    selection_reason: str
    suggested_audio_action: str


@dataclass(frozen=True)
class BenchmarkAuditCandidate:
    row: ManifestRow
    audio_stats: AudioStats
    wer: float
    cer: float
    bucket_flags: str
    reference: str
    prediction: str
    selection_reason: str
    suggested_audio_action: str


@dataclass(frozen=True)
class BenchmarkAuditBuildResult:
    candidates: list[BenchmarkAuditCandidate]
    skipped_manual_review_ids: list[str]
    skipped_error_analysis_ids: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare isolated audit files and low-volume analysis before "
            "running Phase 05 Experiment B."
        )
    )
    parser.add_argument("--train-csv", default=str(DEFAULT_TRAIN_CSV))
    parser.add_argument("--valid-csv", default=str(DEFAULT_VALID_CSV))
    parser.add_argument("--test-csv", default=str(DEFAULT_TEST_CSV))
    parser.add_argument("--predictions-csv", default=str(DEFAULT_PREDICTIONS_CSV))
    parser.add_argument("--error-analysis-csv", default=str(DEFAULT_ERROR_ANALYSIS_CSV))
    parser.add_argument("--manual-review-csv", default=str(DEFAULT_MANUAL_REVIEW_CSV))
    parser.add_argument("--history-csv", default=str(DEFAULT_HISTORY_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--audit-dir", default=str(DEFAULT_AUDIT_DIR))
    parser.add_argument(
        "--short-clip-threshold",
        type=float,
        default=DEFAULT_SHORT_CLIP_THRESHOLD,
    )
    parser.add_argument(
        "--training-audit-limit",
        type=int,
        default=DEFAULT_TRAINING_AUDIT_LIMIT,
    )
    parser.add_argument(
        "--benchmark-audit-limit",
        type=int,
        default=DEFAULT_BENCHMARK_AUDIT_LIMIT,
    )
    return parser.parse_args()


def load_manifest(path: Path, split: str) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw_row in csv.DictReader(handle):
            sample_id = str(raw_row["id"])
            rows.append(
                ManifestRow(
                    split=split,
                    sample_id=sample_id,
                    text=str(raw_row["text"]),
                    duration_seconds=float(raw_row["duration"]),
                    wav_path=f"dataset/extracted_wavs/{sample_id}.wav",
                )
            )
    return rows


def amplitude_to_dbfs(value: float) -> float:
    if value <= 0:
        return -120.0
    return 20.0 * math.log10(value)


def compute_audio_stats(wav_path: Path) -> AudioStats:
    audio, _ = sf.read(str(wav_path), always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    if len(audio) == 0:
        return AudioStats(rms_dbfs=-120.0, peak_dbfs=-120.0)
    abs_audio = abs(audio)
    peak = float(abs_audio.max())
    rms = float((abs_audio**2).mean() ** 0.5)
    return AudioStats(
        rms_dbfs=amplitude_to_dbfs(rms),
        peak_dbfs=amplitude_to_dbfs(peak),
    )


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("Cannot compute percentile from an empty list.")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def suggested_audio_action(audio_stats: AudioStats, low_volume_dbfs: float) -> str:
    if audio_stats.rms_dbfs <= low_volume_dbfs and audio_stats.peak_dbfs <= -6.0:
        return "consider_gain_normalize"
    if audio_stats.rms_dbfs <= low_volume_dbfs:
        return "review_before_gain_change"
    return "keep_raw"


def build_training_audit_candidates(
    rows: list[ManifestRow],
    audio_stats_by_id: dict[str, AudioStats],
    short_clip_threshold: float,
    low_volume_dbfs: float,
    very_low_volume_dbfs: float,
    limit: int,
) -> list[TrainingAuditCandidate]:
    candidates: list[TrainingAuditCandidate] = []
    for row in rows:
        audio_stats = audio_stats_by_id[row.sample_id]
        is_low_volume = audio_stats.rms_dbfs <= low_volume_dbfs
        is_very_low_volume = audio_stats.rms_dbfs <= very_low_volume_dbfs
        has_latin_text = has_latin(row.text)
        is_short_clip = row.duration_seconds < short_clip_threshold
        should_include = is_very_low_volume or (is_low_volume and (has_latin_text or is_short_clip))
        if not should_include:
            continue

        reason_parts: list[str] = []
        if is_very_low_volume:
            reason_parts.append("very_low_volume")
        elif is_low_volume:
            reason_parts.append("low_volume")
        if has_latin_text:
            reason_parts.append("code_switched")
        if is_short_clip:
            reason_parts.append("short_clip")

        candidates.append(
            TrainingAuditCandidate(
                row=row,
                audio_stats=audio_stats,
                has_latin_text=has_latin_text,
                is_short_clip=is_short_clip,
                selection_reason="|".join(reason_parts),
                suggested_audio_action=suggested_audio_action(audio_stats, low_volume_dbfs),
            )
        )

    return sorted(
        candidates,
        key=lambda item: (item.audio_stats.rms_dbfs, item.row.duration_seconds),
    )[:limit]


def load_prediction_metrics(path: Path) -> dict[str, dict[str, str]]:
    metrics: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            metrics[str(row["id"])] = row
    return metrics


def build_benchmark_audit_candidates(
    test_rows: list[ManifestRow],
    audio_stats_by_id: dict[str, AudioStats],
    manual_review_rows: list[dict[str, str]],
    error_analysis_rows: dict[str, dict[str, str]],
    short_clip_threshold: float,
    low_volume_dbfs: float,
    limit: int,
) -> BenchmarkAuditBuildResult:
    rows_by_id = {row.sample_id: row for row in test_rows}
    selected_ids: list[str] = []
    skipped_manual_review_ids: list[str] = []
    manual_review_ids = {str(item["id"]) for item in manual_review_rows}

    for row in manual_review_rows:
        sample_id = str(row["id"])
        if sample_id not in rows_by_id:
            skipped_manual_review_ids.append(sample_id)
            continue
        if sample_id not in selected_ids:
            selected_ids.append(sample_id)

    skipped_error_analysis_ids = [
        sample_id for sample_id in error_analysis_rows if sample_id not in rows_by_id
    ]
    low_volume_error_ids = [
        sample_id
        for sample_id, metrics in error_analysis_rows.items()
        if sample_id in rows_by_id
        and audio_stats_by_id[sample_id].rms_dbfs <= low_volume_dbfs
        and float(metrics["wer"]) >= 0.5
    ]
    for sample_id in sorted(
        low_volume_error_ids,
        key=lambda item: (
            audio_stats_by_id[item].rms_dbfs,
            -float(error_analysis_rows[item]["wer"]),
        ),
    ):
        if sample_id not in selected_ids:
            selected_ids.append(sample_id)
        if len(selected_ids) >= limit:
            break

    candidates: list[BenchmarkAuditCandidate] = []
    for sample_id in selected_ids[:limit]:
        row = rows_by_id.get(sample_id)
        metrics = error_analysis_rows.get(sample_id)
        if row is None or metrics is None:
            continue
        audio_stats = audio_stats_by_id[sample_id]
        reasons: list[str] = []
        if sample_id in manual_review_ids:
            reasons.append("manual_review_candidate")
        if audio_stats.rms_dbfs <= low_volume_dbfs and float(metrics["wer"]) >= 0.5:
            reasons.append("low_volume_bad_prediction")
        if row.duration_seconds < short_clip_threshold:
            reasons.append("short_clip")
        if has_latin(row.text):
            reasons.append("code_switched")

        candidates.append(
            BenchmarkAuditCandidate(
                row=row,
                audio_stats=audio_stats,
                wer=float(metrics["wer"]),
                cer=float(metrics["cer"]),
                bucket_flags=str(metrics["bucket_flags"]),
                reference=str(metrics["reference"]),
                prediction=str(metrics["prediction"]),
                selection_reason="|".join(reasons),
                suggested_audio_action=suggested_audio_action(audio_stats, low_volume_dbfs),
            )
        )
    return BenchmarkAuditBuildResult(
        candidates=candidates,
        skipped_manual_review_ids=skipped_manual_review_ids,
        skipped_error_analysis_ids=skipped_error_analysis_ids,
    )


def write_training_audit(path: Path, candidates: list[TrainingAuditCandidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "id",
                "wav_path",
                "duration_seconds",
                "text",
                "has_latin",
                "is_short_clip",
                "rms_dbfs",
                "peak_dbfs",
                "selection_reason",
                "suggested_audio_action",
                "audit_status",
                "transcript_action",
                "audio_action",
                "keep_for_phase05_boost",
                "corrected_text",
                "notes",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(
                {
                    "split": candidate.row.split,
                    "id": candidate.row.sample_id,
                    "wav_path": candidate.row.wav_path,
                    "duration_seconds": f"{candidate.row.duration_seconds:.3f}",
                    "text": candidate.row.text,
                    "has_latin": str(candidate.has_latin_text),
                    "is_short_clip": str(candidate.is_short_clip),
                    "rms_dbfs": f"{candidate.audio_stats.rms_dbfs:.3f}",
                    "peak_dbfs": f"{candidate.audio_stats.peak_dbfs:.3f}",
                    "selection_reason": candidate.selection_reason,
                    "suggested_audio_action": candidate.suggested_audio_action,
                    "audit_status": "",
                    "transcript_action": "",
                    "audio_action": "",
                    "keep_for_phase05_boost": "",
                    "corrected_text": "",
                    "notes": "",
                }
            )


def write_benchmark_audit(path: Path, candidates: list[BenchmarkAuditCandidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "id",
                "wav_path",
                "duration_seconds",
                "text",
                "rms_dbfs",
                "peak_dbfs",
                "wer",
                "cer",
                "bucket_flags",
                "selection_reason",
                "suggested_audio_action",
                "reference",
                "prediction",
                "benchmark_audit_status",
                "reference_action",
                "corrected_reference",
                "notes",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(
                {
                    "split": candidate.row.split,
                    "id": candidate.row.sample_id,
                    "wav_path": candidate.row.wav_path,
                    "duration_seconds": f"{candidate.row.duration_seconds:.3f}",
                    "text": candidate.row.text,
                    "rms_dbfs": f"{candidate.audio_stats.rms_dbfs:.3f}",
                    "peak_dbfs": f"{candidate.audio_stats.peak_dbfs:.3f}",
                    "wer": f"{candidate.wer:.6f}",
                    "cer": f"{candidate.cer:.6f}",
                    "bucket_flags": candidate.bucket_flags,
                    "selection_reason": candidate.selection_reason,
                    "suggested_audio_action": candidate.suggested_audio_action,
                    "reference": candidate.reference,
                    "prediction": candidate.prediction,
                    "benchmark_audit_status": "",
                    "reference_action": "",
                    "corrected_reference": "",
                    "notes": "",
                }
            )


def write_audio_inventory(
    path: Path,
    rows: list[ManifestRow],
    audio_stats_by_id: dict[str, AudioStats],
    short_clip_threshold: float,
    low_volume_dbfs: float,
    very_low_volume_dbfs: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "id",
                "wav_path",
                "duration_seconds",
                "text",
                "has_latin",
                "is_short_clip",
                "rms_dbfs",
                "peak_dbfs",
                "volume_bucket",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            audio_stats = audio_stats_by_id[row.sample_id]
            volume_bucket = "normal"
            if audio_stats.rms_dbfs <= very_low_volume_dbfs:
                volume_bucket = "very_low"
            elif audio_stats.rms_dbfs <= low_volume_dbfs:
                volume_bucket = "low"
            writer.writerow(
                {
                    "split": row.split,
                    "id": row.sample_id,
                    "wav_path": row.wav_path,
                    "duration_seconds": f"{row.duration_seconds:.3f}",
                    "text": row.text,
                    "has_latin": str(has_latin(row.text)),
                    "is_short_clip": str(row.duration_seconds < short_clip_threshold),
                    "rms_dbfs": f"{audio_stats.rms_dbfs:.3f}",
                    "peak_dbfs": f"{audio_stats.peak_dbfs:.3f}",
                    "volume_bucket": volume_bucket,
                }
            )


def build_freeze_markdown(
    history_path: Path,
    reference_run_name: str,
    frozen_run_name: str,
) -> str:
    rows_by_name: dict[str, dict[str, str]] = {}
    with history_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows_by_name[str(row["run_name"])] = row

    reference_row = rows_by_name[reference_run_name]
    frozen_row = rows_by_name[frozen_run_name]
    wer_delta = float(frozen_row["wer"]) - float(reference_row["wer"])
    cer_delta = float(frozen_row["cer"]) - float(reference_row["cer"])

    return "\n".join(
        [
            "# Phase 05 Experiment A Freeze",
            "",
            "This file freezes the current best decode-only result before Experiment B.",
            "",
            "## Current Winner",
            "",
            f"- Frozen run: `{frozen_run_name}`",
            "- Frozen decoding preset: `phase05_safe_decode_v1`",
            "- Status: use this as the current best inference policy",
            "",
            "## Locked Test Comparison",
            "",
            f"- Reference locked run: `{reference_run_name}`",
            f"- Reference WER: `{float(reference_row['wer']):.6f}`",
            f"- Reference CER: `{float(reference_row['cer']):.6f}`",
            f"- Frozen WER: `{float(frozen_row['wer']):.6f}`",
            f"- Frozen CER: `{float(frozen_row['cer']):.6f}`",
            f"- WER delta: `{wer_delta:+.6f}`",
            f"- CER delta: `{cer_delta:+.6f}`",
            "",
            "## Experiment B Isolation Rule",
            "",
            (
                "Train and evaluate the first data-only Experiment B run with "
                "`--decoding-preset standard` so the data effect stays isolated."
            ),
            "",
            (
                "Only after the data-only result is measured should the frozen "
                "safe decode preset be combined with the new checkpoint."
            ),
            "",
        ]
    )


def build_summary_markdown(
    train_rows: list[ManifestRow],
    valid_rows: list[ManifestRow],
    test_rows: list[ManifestRow],
    low_volume_dbfs: float,
    very_low_volume_dbfs: float,
    training_candidates: list[TrainingAuditCandidate],
    benchmark_candidates: list[BenchmarkAuditCandidate],
    skipped_manual_review_ids: list[str],
    skipped_error_analysis_ids: list[str],
) -> str:
    def count_rows(rows: list[ManifestRow], threshold: float) -> int:
        return sum(1 for row in rows if audio_stats_by_id[row.sample_id].rms_dbfs <= threshold)

    return "\n".join(
        [
            "# Phase 05 Experiment B Preparation",
            "",
            "This prep step isolates training-data cleanup from benchmark corrections.",
            "",
            "## Low-Volume Thresholds",
            "",
            (
                "- Low-volume RMS threshold: "
                f"`{low_volume_dbfs:.3f} dBFS` (training-split bottom decile)"
            ),
            (
                "- Very-low-volume RMS threshold: "
                f"`{very_low_volume_dbfs:.3f} dBFS` (training-split bottom 3%)"
            ),
            "",
            "## Split Counts Below Threshold",
            "",
            f"- Train low-volume rows: `{count_rows(train_rows, low_volume_dbfs)}`",
            f"- Train very-low-volume rows: `{count_rows(train_rows, very_low_volume_dbfs)}`",
            f"- Valid low-volume rows: `{count_rows(valid_rows, low_volume_dbfs)}`",
            f"- Test low-volume rows: `{count_rows(test_rows, low_volume_dbfs)}`",
            "",
            "## Outputs",
            "",
            ("- Training audit sheet: `dataset/audits/phase05_experiment_b_training_audit.csv`"),
            ("- Benchmark review sheet: `dataset/audits/phase05_benchmark_reference_audit.csv`"),
            (
                "- Full loudness inventory: "
                "`reports/phase05_preparation/audio_inventory_all_splits.csv`"
            ),
            ("- Experiment A freeze: `reports/phase05_preparation/experiment_a_freeze.md`"),
            "",
            "## Rules",
            "",
            "- Use the training audit sheet to decide what may be boosted in Experiment B.",
            (
                "- Keep validation and test corrections separate in the benchmark "
                "review sheet. Do not feed them into the training manifest builder."
            ),
            (
                "- Only mark `audio_action=gain_normalize` when the speech is still "
                "intelligible and the reference is trustworthy."
            ),
            (
                "- Mark unusable rows for exclusion instead of boosting noise-heavy "
                "or transcript-unclear clips."
            ),
            "",
            "## Current Candidate Counts",
            "",
            f"- Training audit candidates: `{len(training_candidates)}`",
            f"- Benchmark review candidates: `{len(benchmark_candidates)}`",
            f"- Stale manual-review ids skipped: `{len(skipped_manual_review_ids)}`",
            f"- Stale error-analysis ids skipped: `{len(skipped_error_analysis_ids)}`",
            "",
        ]
    )


def load_manual_review_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    train_rows = load_manifest(Path(args.train_csv), split="train")
    valid_rows = load_manifest(Path(args.valid_csv), split="valid")
    test_rows = load_manifest(Path(args.test_csv), split="test")
    all_rows = train_rows + valid_rows + test_rows

    global audio_stats_by_id
    audio_stats_by_id = {
        row.sample_id: compute_audio_stats(ROOT_DIR / row.wav_path) for row in all_rows
    }
    train_rms_values = [audio_stats_by_id[row.sample_id].rms_dbfs for row in train_rows]
    low_volume_dbfs = percentile(train_rms_values, 0.10)
    very_low_volume_dbfs = percentile(train_rms_values, 0.03)

    training_candidates = build_training_audit_candidates(
        rows=train_rows,
        audio_stats_by_id=audio_stats_by_id,
        short_clip_threshold=args.short_clip_threshold,
        low_volume_dbfs=low_volume_dbfs,
        very_low_volume_dbfs=very_low_volume_dbfs,
        limit=args.training_audit_limit,
    )
    benchmark_result = build_benchmark_audit_candidates(
        test_rows=test_rows,
        audio_stats_by_id=audio_stats_by_id,
        manual_review_rows=load_manual_review_rows(Path(args.manual_review_csv)),
        error_analysis_rows=load_prediction_metrics(Path(args.error_analysis_csv)),
        short_clip_threshold=args.short_clip_threshold,
        low_volume_dbfs=low_volume_dbfs,
        limit=args.benchmark_audit_limit,
    )
    benchmark_candidates = benchmark_result.candidates

    output_dir = Path(args.output_dir)
    audit_dir = Path(args.audit_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    write_training_audit(
        audit_dir / "phase05_experiment_b_training_audit.csv",
        training_candidates,
    )
    write_benchmark_audit(
        audit_dir / "phase05_benchmark_reference_audit.csv",
        benchmark_candidates,
    )
    write_audio_inventory(
        output_dir / "audio_inventory_all_splits.csv",
        all_rows,
        audio_stats_by_id,
        short_clip_threshold=args.short_clip_threshold,
        low_volume_dbfs=low_volume_dbfs,
        very_low_volume_dbfs=very_low_volume_dbfs,
    )
    (output_dir / "experiment_a_freeze.md").write_text(
        build_freeze_markdown(
            history_path=Path(args.history_csv),
            reference_run_name="whisper-small-phase03-full-20260428-001840-locked-test",
            frozen_run_name=(
                "whisper-small-phase03-full-20260428-001840-phase05-safe-decode-locked"
            ),
        ),
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(
        build_summary_markdown(
            train_rows=train_rows,
            valid_rows=valid_rows,
            test_rows=test_rows,
            low_volume_dbfs=low_volume_dbfs,
            very_low_volume_dbfs=very_low_volume_dbfs,
            training_candidates=training_candidates,
            benchmark_candidates=benchmark_candidates,
            skipped_manual_review_ids=benchmark_result.skipped_manual_review_ids,
            skipped_error_analysis_ids=benchmark_result.skipped_error_analysis_ids,
        ),
        encoding="utf-8",
    )

    print(f"Saved prep summary to {output_dir / 'summary.md'}")
    print(f"Saved audio inventory to {output_dir / 'audio_inventory_all_splits.csv'}")
    print(f"Saved training audit sheet to {audit_dir / 'phase05_experiment_b_training_audit.csv'}")
    print(f"Saved benchmark review sheet to {audit_dir / 'phase05_benchmark_reference_audit.csv'}")
    if benchmark_result.skipped_manual_review_ids:
        print(f"Skipped stale manual-review ids: {len(benchmark_result.skipped_manual_review_ids)}")
    if benchmark_result.skipped_error_analysis_ids:
        print(
            f"Skipped stale error-analysis ids: {len(benchmark_result.skipped_error_analysis_ids)}"
        )


if __name__ == "__main__":
    main()
