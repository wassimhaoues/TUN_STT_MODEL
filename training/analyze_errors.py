from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training.baseline_test import format_metric  # noqa: E402

REPORTS_DIR = ROOT_DIR / "reports"
DEFAULT_SOURCE_CSV = ROOT_DIR / "dataset" / "metadata_test.csv"

ARABIC_PATTERN = re.compile(r"[\u0600-\u06FF]")
LATIN_PATTERN = re.compile(r"[A-Za-z]")
SHORT_CLIP_SECONDS = 3.0
LONG_CLIP_SECONDS = 10.0
HIGH_CER_THRESHOLD = 0.35
MODERATE_WER_MIN = 0.2
MODERATE_WER_MAX = 0.6
OMISSION_TOKEN_RATIO = 0.6
OMISSION_WER_THRESHOLD = 0.5
REPEATED_WER_THRESHOLD = 0.75
CATASTROPHIC_WER_THRESHOLD = 2.0
MANUAL_REVIEW_LIMIT = 20
CER_GOOD_MAX = 0.15
CER_ACCEPTABLE_MAX = 0.25
CER_NEEDS_REVIEW_MAX = 0.40

CER_REVIEW_BAND_ORDER = (
    "good",
    "acceptable",
    "needs_review",
    "high_priority_review",
    "critical",
)

CER_REVIEW_BAND_LABELS = {
    "good": "Good",
    "acceptable": "Acceptable but inspect sometimes",
    "needs_review": "Needs review",
    "high_priority_review": "High priority review",
    "critical": "Critical",
}

BUCKET_ORDER = (
    "code_switched_reference",
    "arabic_only_reference",
    "short_clip",
    "long_clip",
    "high_cer_moderate_wer",
    "major_omission",
    "repeated_token_hallucination",
    "catastrophic_looping",
)

BUCKET_DESCRIPTIONS = {
    "code_switched_reference": "Reference contains Latin-script code-switching.",
    "arabic_only_reference": "Reference is Arabic-only with no Latin-script words.",
    "short_clip": f"Clip duration is shorter than {SHORT_CLIP_SECONDS:.1f} seconds.",
    "long_clip": f"Clip duration is at least {LONG_CLIP_SECONDS:.1f} seconds.",
    "high_cer_moderate_wer": "Character corruption is high even when word error is only moderate.",
    "major_omission": "Prediction is much shorter than the reference and misses large content.",
    "repeated_token_hallucination": (
        "Prediction contains repeated token loops not present in the reference."
    ),
    "catastrophic_looping": "A severe repetition loop causes extremely large WER.",
}


@dataclass(frozen=True)
class PredictionSample:
    sample_id: str
    wav_path: str
    reference: str
    prediction: str
    duration_seconds: float
    reference_word_count: int
    prediction_word_count: int
    word_edits: int
    char_edits: int
    wer: float
    cer: float
    bucket_flags: tuple[str, ...]
    cer_review_band: str
    critical_review_flag: bool
    critical_review_reason: str


@dataclass(frozen=True)
class BucketSummary:
    bucket_name: str
    description: str
    sample_count: int
    sample_rate: float
    average_wer: float
    average_cer: float
    corpus_wer: float
    corpus_cer: float


@dataclass(frozen=True)
class AnalysisReport:
    prediction_run_name: str
    predictions_csv: str
    source_csv: str
    total_samples: int
    corpus_wer: float
    corpus_cer: float
    short_clip_threshold: float
    long_clip_threshold: float
    bucket_summaries: list[BucketSummary]
    manual_findings: list[str]
    worst_samples: list[PredictionSample]
    manual_review_candidates: list[PredictionSample]
    cer_review_counts: dict[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze ASR prediction errors and generate Phase 04 reports."
    )
    parser.add_argument(
        "--predictions-csv",
        required=True,
        help="Predictions CSV produced by baseline or checkpoint evaluation.",
    )
    parser.add_argument(
        "--source-csv",
        default=str(DEFAULT_SOURCE_CSV),
        help="Source manifest used to recover durations. Defaults to dataset/metadata_test.csv.",
    )
    return parser.parse_args()


def tokenize_words(text: str) -> list[str]:
    return text.split()


def levenshtein_distance(left: list[str], right: list[str]) -> int:
    if not left:
        return len(right)
    if not right:
        return len(left)

    distances = list(range(len(right) + 1))
    for left_index, left_token in enumerate(left, start=1):
        previous_diagonal = distances[0]
        distances[0] = left_index
        for right_index, right_token in enumerate(right, start=1):
            current = distances[right_index]
            distances[right_index] = min(
                distances[right_index] + 1,
                distances[right_index - 1] + 1,
                previous_diagonal + int(left_token != right_token),
            )
            previous_diagonal = current
    return distances[-1]


def has_arabic(text: str) -> bool:
    return bool(ARABIC_PATTERN.search(text))


def has_latin(text: str) -> bool:
    return bool(LATIN_PATTERN.search(text))


def max_repetition_run(tokens: list[str]) -> int:
    longest_run = 1 if tokens else 0

    for index, token in enumerate(tokens):
        run_length = 1
        next_index = index + 1
        while next_index < len(tokens) and tokens[next_index] == token:
            run_length += 1
            next_index += 1
        longest_run = max(longest_run, run_length)

    for ngram_size in (2, 3):
        stop = len(tokens) - (2 * ngram_size) + 1
        for index in range(max(stop, 0)):
            current_ngram = tokens[index : index + ngram_size]
            repeat_count = 1
            probe = index + ngram_size
            while probe + ngram_size <= len(tokens):
                next_ngram = tokens[probe : probe + ngram_size]
                if next_ngram != current_ngram:
                    break
                repeat_count += 1
                probe += ngram_size
            longest_run = max(longest_run, repeat_count)

    return longest_run


def has_repetition_loop(tokens: list[str]) -> bool:
    return max_repetition_run(tokens) >= 2


def is_wrong_language_prediction(reference: str, prediction: str) -> bool:
    reference_has_arabic = has_arabic(reference)
    prediction_has_arabic = has_arabic(prediction)
    reference_has_latin = has_latin(reference)
    prediction_has_latin = has_latin(prediction)
    if reference_has_arabic and not prediction_has_arabic:
        return True
    if not reference_has_latin and prediction_has_latin and not prediction_has_arabic:
        return True
    return False


def resolve_critical_review_reason(
    prediction: str,
    repeated_hallucination: bool,
    catastrophic_looping: bool,
    major_omission: bool,
    wrong_language_prediction: bool,
) -> str:
    if not prediction.strip():
        return "empty_prediction"
    if catastrophic_looping:
        return "catastrophic_looping"
    if repeated_hallucination:
        return "repeated_words"
    if wrong_language_prediction:
        return "wrong_language"
    if major_omission:
        return "major_omission"
    return ""


def resolve_cer_review_band(cer: float, critical_review_reason: str) -> str:
    if critical_review_reason:
        return "critical"
    if cer < CER_GOOD_MAX:
        return "good"
    if cer < CER_ACCEPTABLE_MAX:
        return "acceptable"
    if cer < CER_NEEDS_REVIEW_MAX:
        return "needs_review"
    return "high_priority_review"


def analyze_prediction_row(
    prediction_row: dict[str, str],
    source_row: dict[str, str],
) -> PredictionSample:
    reference = prediction_row["reference"].strip()
    prediction = prediction_row["prediction"].strip()
    reference_words = tokenize_words(reference)
    prediction_words = tokenize_words(prediction)

    word_edits = levenshtein_distance(reference_words, prediction_words)
    char_edits = levenshtein_distance(list(reference), list(prediction))
    wer = word_edits / max(1, len(reference_words))
    cer = char_edits / max(1, len(reference))
    duration_seconds = float(source_row["duration"])

    repeated_prediction = has_repetition_loop(prediction_words)
    prediction_repeat_run = max_repetition_run(prediction_words)
    reference_repeat_run = max_repetition_run(reference_words)
    major_omission = (
        len(prediction_words) <= max(1.0, len(reference_words) * OMISSION_TOKEN_RATIO)
        and wer >= OMISSION_WER_THRESHOLD
    )
    repeated_hallucination = (
        repeated_prediction
        and prediction_repeat_run >= max(3, reference_repeat_run + 2)
        and (wer >= REPEATED_WER_THRESHOLD or len(prediction_words) >= len(reference_words) + 4)
    )
    catastrophic_looping = repeated_hallucination and wer >= CATASTROPHIC_WER_THRESHOLD
    wrong_language_prediction = is_wrong_language_prediction(reference, prediction)
    critical_review_reason = resolve_critical_review_reason(
        prediction=prediction,
        repeated_hallucination=repeated_hallucination,
        catastrophic_looping=catastrophic_looping,
        major_omission=major_omission,
        wrong_language_prediction=wrong_language_prediction,
    )
    cer_review_band = resolve_cer_review_band(cer, critical_review_reason)

    bucket_flags: list[str] = []
    if has_latin(reference):
        bucket_flags.append("code_switched_reference")
    if has_arabic(reference) and not has_latin(reference):
        bucket_flags.append("arabic_only_reference")
    if duration_seconds < SHORT_CLIP_SECONDS:
        bucket_flags.append("short_clip")
    if duration_seconds >= LONG_CLIP_SECONDS:
        bucket_flags.append("long_clip")
    if cer >= HIGH_CER_THRESHOLD and MODERATE_WER_MIN <= wer <= MODERATE_WER_MAX:
        bucket_flags.append("high_cer_moderate_wer")
    if major_omission:
        bucket_flags.append("major_omission")
    if repeated_hallucination:
        bucket_flags.append("repeated_token_hallucination")
    if catastrophic_looping:
        bucket_flags.append("catastrophic_looping")

    return PredictionSample(
        sample_id=prediction_row["id"],
        wav_path=prediction_row["wav_path"],
        reference=reference,
        prediction=prediction,
        duration_seconds=duration_seconds,
        reference_word_count=len(reference_words),
        prediction_word_count=len(prediction_words),
        word_edits=word_edits,
        char_edits=char_edits,
        wer=wer,
        cer=cer,
        bucket_flags=tuple(bucket_flags),
        cer_review_band=cer_review_band,
        critical_review_flag=bool(critical_review_reason),
        critical_review_reason=critical_review_reason,
    )


def build_bucket_summary(samples: list[PredictionSample], bucket_name: str) -> BucketSummary:
    bucket_samples = [sample for sample in samples if bucket_name in sample.bucket_flags]
    if not bucket_samples:
        return BucketSummary(
            bucket_name=bucket_name,
            description=BUCKET_DESCRIPTIONS[bucket_name],
            sample_count=0,
            sample_rate=0.0,
            average_wer=float("nan"),
            average_cer=float("nan"),
            corpus_wer=float("nan"),
            corpus_cer=float("nan"),
        )

    total_ref_words = sum(sample.reference_word_count for sample in bucket_samples)
    total_ref_chars = sum(len(sample.reference) for sample in bucket_samples)
    total_word_edits = sum(sample.word_edits for sample in bucket_samples)
    total_char_edits = sum(sample.char_edits for sample in bucket_samples)
    return BucketSummary(
        bucket_name=bucket_name,
        description=BUCKET_DESCRIPTIONS[bucket_name],
        sample_count=len(bucket_samples),
        sample_rate=len(bucket_samples) / max(1, len(samples)),
        average_wer=mean(sample.wer for sample in bucket_samples),
        average_cer=mean(sample.cer for sample in bucket_samples),
        corpus_wer=total_word_edits / max(1, total_ref_words),
        corpus_cer=total_char_edits / max(1, total_ref_chars),
    )


def build_manual_findings(
    samples: list[PredictionSample],
    bucket_map: dict[str, BucketSummary],
) -> list[str]:
    findings: list[str] = []
    code_switch = bucket_map["code_switched_reference"]
    arabic_only = bucket_map["arabic_only_reference"]
    if code_switch.sample_count and arabic_only.sample_count:
        findings.append(
            "Code-switching is still a distinct regime in the test set: "
            f"average WER is {format_metric(code_switch.average_wer)} on "
            f"{code_switch.sample_count} code-switched samples versus "
            f"{format_metric(arabic_only.average_wer)} on "
            f"{arabic_only.sample_count} Arabic-only samples."
        )

    short_clip = bucket_map["short_clip"]
    long_clip = bucket_map["long_clip"]
    if short_clip.sample_count and long_clip.sample_count:
        findings.append(
            "Duration still matters: "
            f"short clips under {SHORT_CLIP_SECONDS:.1f}s reach corpus WER "
            f"{format_metric(short_clip.corpus_wer)}, while clips at or above "
            f"{LONG_CLIP_SECONDS:.1f}s reach {format_metric(long_clip.corpus_wer)}."
        )

    repeated = bucket_map["repeated_token_hallucination"]
    catastrophic = bucket_map["catastrophic_looping"]
    if repeated.sample_count:
        findings.append(
            "A small set of repetition loops is driving the worst failures: "
            f"{repeated.sample_count} samples show repeated-token hallucination, "
            f"including {catastrophic.sample_count} catastrophic loops with WER "
            f"above {CATASTROPHIC_WER_THRESHOLD:.1f}."
        )

    omission = bucket_map["major_omission"]
    if omission.sample_count:
        findings.append(
            "Major omissions are rare but real: "
            f"{omission.sample_count} samples collapse into much shorter "
            "predictions, often on very short or Latin-heavy references."
        )

    high_cer = bucket_map["high_cer_moderate_wer"]
    if high_cer.sample_count:
        findings.append(
            "Some errors preserve the rough word skeleton but corrupt token forms: "
            f"{high_cer.sample_count} samples land in the high-CER / moderate-WER bucket."
        )

    if not findings:
        findings.append("No dominant failure bucket crossed the current heuristics.")

    return findings


def select_worst_samples(
    samples: list[PredictionSample],
    limit: int = 10,
) -> list[PredictionSample]:
    return sorted(
        samples,
        key=lambda sample: (sample.wer, sample.cer, sample.duration_seconds),
        reverse=True,
    )[:limit]


def select_manual_review_candidates(samples: list[PredictionSample]) -> list[PredictionSample]:
    candidate_buckets = (
        "catastrophic_looping",
        "repeated_token_hallucination",
        "major_omission",
        "high_cer_moderate_wer",
        "long_clip",
        "short_clip",
        "code_switched_reference",
    )
    selected: list[PredictionSample] = []
    seen_ids: set[str] = set()
    for bucket_name in candidate_buckets:
        bucket_samples = [sample for sample in samples if bucket_name in sample.bucket_flags]
        for sample in sorted(bucket_samples, key=lambda item: (item.wer, item.cer), reverse=True):
            if sample.sample_id in seen_ids:
                continue
            selected.append(sample)
            seen_ids.add(sample.sample_id)
            if len(selected) >= MANUAL_REVIEW_LIMIT:
                return selected
            if sum(1 for item in selected if bucket_name in item.bucket_flags) >= 3:
                break
    return selected


def build_cer_review_counts(samples: list[PredictionSample]) -> dict[str, int]:
    return {
        band: sum(1 for sample in samples if sample.cer_review_band == band)
        for band in CER_REVIEW_BAND_ORDER
    }


def select_cer_review_queue(samples: list[PredictionSample]) -> list[PredictionSample]:
    band_priority = {
        "critical": 0,
        "high_priority_review": 1,
        "needs_review": 2,
        "acceptable": 3,
        "good": 4,
    }
    return sorted(
        samples,
        key=lambda sample: (
            band_priority[sample.cer_review_band],
            -sample.cer,
            -sample.wer,
            -sample.duration_seconds,
        ),
    )


def build_analysis_report(
    predictions_csv: Path,
    source_csv: Path,
    samples: list[PredictionSample],
) -> AnalysisReport:
    total_ref_words = sum(sample.reference_word_count for sample in samples)
    total_ref_chars = sum(len(sample.reference) for sample in samples)
    total_word_edits = sum(sample.word_edits for sample in samples)
    total_char_edits = sum(sample.char_edits for sample in samples)
    bucket_summaries = [build_bucket_summary(samples, bucket_name) for bucket_name in BUCKET_ORDER]
    bucket_map = {bucket.bucket_name: bucket for bucket in bucket_summaries}

    try:
        predictions_csv_display = str(predictions_csv.relative_to(ROOT_DIR))
    except ValueError:
        predictions_csv_display = str(predictions_csv)

    try:
        source_csv_display = str(source_csv.relative_to(ROOT_DIR))
    except ValueError:
        source_csv_display = str(source_csv)

    return AnalysisReport(
        prediction_run_name=predictions_csv.parent.name,
        predictions_csv=predictions_csv_display,
        source_csv=source_csv_display,
        total_samples=len(samples),
        corpus_wer=total_word_edits / max(1, total_ref_words),
        corpus_cer=total_char_edits / max(1, total_ref_chars),
        short_clip_threshold=SHORT_CLIP_SECONDS,
        long_clip_threshold=LONG_CLIP_SECONDS,
        bucket_summaries=bucket_summaries,
        manual_findings=build_manual_findings(samples, bucket_map),
        worst_samples=select_worst_samples(samples),
        manual_review_candidates=select_manual_review_candidates(samples),
        cer_review_counts=build_cer_review_counts(samples),
    )


def build_summary_markdown(report: AnalysisReport) -> str:
    bucket_lines = []
    for bucket in report.bucket_summaries:
        bucket_lines.extend(
            [
                f"### {bucket.bucket_name}",
                "",
                f"- Description: {bucket.description}",
                f"- Sample count: `{bucket.sample_count}` / `{report.total_samples}` "
                f"(`{bucket.sample_rate * 100:.2f}%`)",
                f"- Average WER: `{format_metric(bucket.average_wer)}`",
                f"- Average CER: `{format_metric(bucket.average_cer)}`",
                f"- Corpus WER: `{format_metric(bucket.corpus_wer)}`",
                f"- Corpus CER: `{format_metric(bucket.corpus_cer)}`",
                "",
            ]
        )

    worst_lines = []
    for sample in report.worst_samples[:10]:
        worst_lines.extend(
            [
                f"- `{sample.sample_id}` | duration `{sample.duration_seconds:.3f}` | "
                f"WER `{format_metric(sample.wer)}` | CER `{format_metric(sample.cer)}` | "
                f"flags `{', '.join(sample.bucket_flags) or 'none'}`",
                f"  ref: {sample.reference}",
                f"  pred: {sample.prediction}",
            ]
        )

    findings_lines = [f"- {finding}" for finding in report.manual_findings]

    return "\n".join(
        [
            f"# Phase 04 Error Analysis: {report.prediction_run_name}",
            "",
            (
                "This report turns prediction-level outputs into concrete failure "
                "buckets for follow-up training decisions."
            ),
            "",
            "## Inputs",
            "",
            f"- Predictions CSV: `{report.predictions_csv}`",
            f"- Source manifest: `{report.source_csv}`",
            f"- Total samples: `{report.total_samples}`",
            "",
            "## Overall Metrics",
            "",
            f"- Corpus WER: `{format_metric(report.corpus_wer)}`",
            f"- Corpus CER: `{format_metric(report.corpus_cer)}`",
            f"- Short clip threshold: `< {report.short_clip_threshold:.1f}s`",
            f"- Long clip threshold: `>= {report.long_clip_threshold:.1f}s`",
            "",
            "## Main Findings",
            "",
            *findings_lines,
            "",
            "## CER Review Bands",
            "",
            *[
                f"- {CER_REVIEW_BAND_LABELS[band]}: `{report.cer_review_counts[band]}`"
                for band in CER_REVIEW_BAND_ORDER
            ],
            "",
            "## Bucket Summary",
            "",
            *bucket_lines,
            "## Worst Samples",
            "",
            *worst_lines,
            "",
            "## Manual Review Set",
            "",
            (
                "See `manual_review_candidates.csv` for a fixed set of high-value "
                "examples to inspect by hand."
            ),
            "",
        ]
    )


def write_bucket_summary_csv(path: Path, bucket_summaries: list[BucketSummary]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "bucket_name",
                "description",
                "sample_count",
                "sample_rate",
                "average_wer",
                "average_cer",
                "corpus_wer",
                "corpus_cer",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for bucket in bucket_summaries:
            writer.writerow(
                {
                    "bucket_name": bucket.bucket_name,
                    "description": bucket.description,
                    "sample_count": bucket.sample_count,
                    "sample_rate": f"{bucket.sample_rate:.6f}",
                    "average_wer": format_metric(bucket.average_wer),
                    "average_cer": format_metric(bucket.average_cer),
                    "corpus_wer": format_metric(bucket.corpus_wer),
                    "corpus_cer": format_metric(bucket.corpus_cer),
                }
            )


def write_sample_csv(path: Path, samples: list[PredictionSample]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "wav_path",
                "duration_seconds",
                "reference_word_count",
                "prediction_word_count",
                "word_edits",
                "char_edits",
                "wer",
                "cer",
                "bucket_flags",
                "cer_review_band",
                "critical_review_flag",
                "critical_review_reason",
                "reference",
                "prediction",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "id": sample.sample_id,
                    "wav_path": sample.wav_path,
                    "duration_seconds": f"{sample.duration_seconds:.3f}",
                    "reference_word_count": sample.reference_word_count,
                    "prediction_word_count": sample.prediction_word_count,
                    "word_edits": sample.word_edits,
                    "char_edits": sample.char_edits,
                    "wer": format_metric(sample.wer),
                    "cer": format_metric(sample.cer),
                    "bucket_flags": "|".join(sample.bucket_flags),
                    "cer_review_band": sample.cer_review_band,
                    "critical_review_flag": str(sample.critical_review_flag),
                    "critical_review_reason": sample.critical_review_reason,
                    "reference": sample.reference,
                    "prediction": sample.prediction,
                }
            )


def write_cer_review_band_csvs(output_dir: Path, samples: list[PredictionSample]) -> list[Path]:
    output_paths: list[Path] = []
    queue_samples = select_cer_review_queue(samples)
    queue_path = output_dir / "cer_review_queue.csv"
    write_sample_csv(queue_path, queue_samples)
    output_paths.append(queue_path)

    for band in CER_REVIEW_BAND_ORDER:
        band_samples = [sample for sample in queue_samples if sample.cer_review_band == band]
        band_path = output_dir / f"cer_{band}.csv"
        write_sample_csv(band_path, band_samples)
        output_paths.append(band_path)

    return output_paths


def load_samples(predictions_csv: Path, source_csv: Path) -> list[PredictionSample]:
    with source_csv.open(encoding="utf-8", newline="") as handle:
        source_rows = {row["id"]: row for row in csv.DictReader(handle)}

    samples: list[PredictionSample] = []
    with predictions_csv.open(encoding="utf-8", newline="") as handle:
        for prediction_row in csv.DictReader(handle):
            source_row = source_rows.get(prediction_row["id"])
            if source_row is None:
                raise KeyError(
                    f"Prediction id {prediction_row['id']} was not found in {source_csv}."
                )
            samples.append(analyze_prediction_row(prediction_row, source_row))
    return samples


def save_analysis_report(
    report: AnalysisReport,
    output_dir: Path,
    samples: list[PredictionSample],
) -> tuple[Path, Path, Path, Path, list[Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.md"
    bucket_summary_path = output_dir / "bucket_summary.csv"
    detailed_rows_path = output_dir / "detailed_rows.csv"
    manual_review_path = output_dir / "manual_review_candidates.csv"

    summary_path.write_text(build_summary_markdown(report), encoding="utf-8")
    write_bucket_summary_csv(bucket_summary_path, report.bucket_summaries)
    write_sample_csv(detailed_rows_path, samples)
    write_sample_csv(manual_review_path, report.manual_review_candidates)
    cer_review_paths = write_cer_review_band_csvs(output_dir, samples)

    return (
        summary_path,
        bucket_summary_path,
        detailed_rows_path,
        manual_review_path,
        cer_review_paths,
    )


def resolve_output_dir(predictions_csv: Path) -> Path:
    if predictions_csv.parent.parent.name == "runs" and predictions_csv.parent.name:
        return predictions_csv.parent / "error_analysis"
    return REPORTS_DIR / "analyses" / predictions_csv.stem


def main() -> None:
    args = parse_args()
    predictions_csv = Path(args.predictions_csv).resolve()
    source_csv = Path(args.source_csv).resolve()
    samples = load_samples(predictions_csv, source_csv)
    report = build_analysis_report(predictions_csv, source_csv, samples)
    output_dir = resolve_output_dir(predictions_csv)
    (
        summary_path,
        bucket_summary_path,
        detailed_rows_path,
        manual_review_path,
        cer_review_paths,
    ) = save_analysis_report(
        report,
        output_dir,
        samples,
    )

    print(f"Saved summary to {summary_path}")
    print(f"Saved bucket summary to {bucket_summary_path}")
    print(f"Saved detailed rows to {detailed_rows_path}")
    print(f"Saved manual review candidates to {manual_review_path}")
    print(f"Saved CER review queue to {cer_review_paths[0]}")


if __name__ == "__main__":
    main()
