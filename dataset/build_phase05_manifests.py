from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DEFAULT_TRAIN_CSV = ROOT_DIR / "dataset" / "metadata_train.csv"
DEFAULT_VALID_CSV = ROOT_DIR / "dataset" / "metadata_valid.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "dataset" / "phase05_manifests"
DEFAULT_REPORTS_DIR = ROOT_DIR / "reports" / "phase05_data_strategies"

LATIN_PATTERN = re.compile(r"[A-Za-z]")
DEFAULT_SHORT_CLIP_THRESHOLD = 3.0


@dataclass(frozen=True)
class ManifestRow:
    id: str
    text: str
    duration: float
    text_raw: str
    normalization_changed: str
    normalization_version: str


@dataclass(frozen=True)
class StrategySummary:
    experiment_name: str
    train_csv: str
    valid_csv: str
    output_train_csv: str
    output_valid_csv: str
    code_switch_boost_factor: int
    short_clip_boost_factor: int
    short_clip_threshold: float
    train_rows_in: int
    train_rows_out: int
    valid_rows_in: int
    valid_rows_out: int
    code_switched_train_rows: int
    short_train_rows: int
    both_code_switched_and_short_rows: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Phase 05 targeted manifests for code-switched and short-clip emphasis."
    )
    parser.add_argument(
        "--experiment-name",
        required=True,
        help="Stable name for this Phase 05 data strategy.",
    )
    parser.add_argument(
        "--train-csv",
        default=str(DEFAULT_TRAIN_CSV),
        help="Input training manifest CSV.",
    )
    parser.add_argument(
        "--valid-csv",
        default=str(DEFAULT_VALID_CSV),
        help="Input validation manifest CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where generated manifests will be written.",
    )
    parser.add_argument(
        "--reports-dir",
        default=str(DEFAULT_REPORTS_DIR),
        help="Directory where the strategy summary will be written.",
    )
    parser.add_argument(
        "--code-switch-boost-factor",
        type=int,
        default=1,
        help="Duplicate code-switched training rows up to this factor. Use 1 to disable.",
    )
    parser.add_argument(
        "--short-clip-boost-factor",
        type=int,
        default=1,
        help="Duplicate short-clip training rows up to this factor. Use 1 to disable.",
    )
    parser.add_argument(
        "--short-clip-threshold",
        type=float,
        default=DEFAULT_SHORT_CLIP_THRESHOLD,
        help=(
            "Seconds threshold for short-clip emphasis. "
            f"Defaults to {DEFAULT_SHORT_CLIP_THRESHOLD}."
        ),
    )
    return parser.parse_args()


def has_latin(text: str) -> bool:
    return bool(LATIN_PATTERN.search(text))


def load_manifest_rows(path: Path) -> list[ManifestRow]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            ManifestRow(
                id=str(row["id"]),
                text=str(row["text"]),
                duration=float(row["duration"]),
                text_raw=str(row.get("text_raw", row["text"])),
                normalization_changed=str(row.get("normalization_changed", "False")),
                normalization_version=str(row.get("normalization_version", "unknown")),
            )
            for row in reader
        ]


def write_manifest_rows(path: Path, rows: list[ManifestRow]) -> None:
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
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def expand_training_rows(
    rows: list[ManifestRow],
    code_switch_boost_factor: int,
    short_clip_boost_factor: int,
    short_clip_threshold: float,
) -> list[ManifestRow]:
    expanded: list[ManifestRow] = []
    for row in rows:
        copies = 1
        if has_latin(row.text):
            copies = max(copies, code_switch_boost_factor)
        if row.duration < short_clip_threshold:
            copies = max(copies, short_clip_boost_factor)
        expanded.extend([row] * copies)
    return expanded


def build_summary(
    experiment_name: str,
    train_csv: Path,
    valid_csv: Path,
    output_train_csv: Path,
    output_valid_csv: Path,
    train_rows: list[ManifestRow],
    expanded_train_rows: list[ManifestRow],
    valid_rows: list[ManifestRow],
    code_switch_boost_factor: int,
    short_clip_boost_factor: int,
    short_clip_threshold: float,
) -> StrategySummary:
    code_switched_train_rows = sum(1 for row in train_rows if has_latin(row.text))
    short_train_rows = sum(1 for row in train_rows if row.duration < short_clip_threshold)
    both_rows = sum(
        1 for row in train_rows if has_latin(row.text) and row.duration < short_clip_threshold
    )
    return StrategySummary(
        experiment_name=experiment_name,
        train_csv=str(train_csv),
        valid_csv=str(valid_csv),
        output_train_csv=str(output_train_csv),
        output_valid_csv=str(output_valid_csv),
        code_switch_boost_factor=code_switch_boost_factor,
        short_clip_boost_factor=short_clip_boost_factor,
        short_clip_threshold=short_clip_threshold,
        train_rows_in=len(train_rows),
        train_rows_out=len(expanded_train_rows),
        valid_rows_in=len(valid_rows),
        valid_rows_out=len(valid_rows),
        code_switched_train_rows=code_switched_train_rows,
        short_train_rows=short_train_rows,
        both_code_switched_and_short_rows=both_rows,
    )


def build_summary_markdown(summary: StrategySummary) -> str:
    return "\n".join(
        [
            f"# Phase 05 Data Strategy: {summary.experiment_name}",
            "",
            "This strategy boosts targeted training examples without changing the validation set.",
            "",
            "## Inputs",
            "",
            f"- Train manifest: `{summary.train_csv}`",
            f"- Valid manifest: `{summary.valid_csv}`",
            "",
            "## Strategy",
            "",
            f"- Code-switch boost factor: `{summary.code_switch_boost_factor}`",
            f"- Short-clip boost factor: `{summary.short_clip_boost_factor}`",
            f"- Short-clip threshold: `{summary.short_clip_threshold:.3f}` seconds",
            "",
            "## Counts",
            "",
            f"- Train rows in: `{summary.train_rows_in}`",
            f"- Train rows out: `{summary.train_rows_out}`",
            f"- Valid rows in: `{summary.valid_rows_in}`",
            f"- Valid rows out: `{summary.valid_rows_out}`",
            f"- Code-switched train rows: `{summary.code_switched_train_rows}`",
            f"- Short train rows: `{summary.short_train_rows}`",
            (
                "- Code-switched and short train rows: "
                f"`{summary.both_code_switched_and_short_rows}`"
            ),
            "",
            "## Outputs",
            "",
            f"- Output train manifest: `{summary.output_train_csv}`",
            f"- Output valid manifest: `{summary.output_valid_csv}`",
            "",
        ]
    )


def save_strategy_report(reports_dir: Path, summary: StrategySummary) -> None:
    strategy_dir = reports_dir / summary.experiment_name
    strategy_dir.mkdir(parents=True, exist_ok=True)
    (strategy_dir / "summary.md").write_text(
        build_summary_markdown(summary),
        encoding="utf-8",
    )
    (strategy_dir / "summary.json").write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.code_switch_boost_factor <= 0:
        raise ValueError("--code-switch-boost-factor must be greater than 0.")
    if args.short_clip_boost_factor <= 0:
        raise ValueError("--short-clip-boost-factor must be greater than 0.")
    if args.short_clip_threshold <= 0:
        raise ValueError("--short-clip-threshold must be greater than 0.")

    train_csv = Path(args.train_csv).resolve()
    valid_csv = Path(args.valid_csv).resolve()
    output_dir = Path(args.output_dir).resolve() / args.experiment_name
    reports_dir = Path(args.reports_dir).resolve()

    train_rows = load_manifest_rows(train_csv)
    valid_rows = load_manifest_rows(valid_csv)
    expanded_train_rows = expand_training_rows(
        train_rows=train_rows,
        code_switch_boost_factor=args.code_switch_boost_factor,
        short_clip_boost_factor=args.short_clip_boost_factor,
        short_clip_threshold=args.short_clip_threshold,
    )

    output_train_csv = output_dir / "metadata_train.csv"
    output_valid_csv = output_dir / "metadata_valid.csv"
    write_manifest_rows(output_train_csv, expanded_train_rows)
    write_manifest_rows(output_valid_csv, valid_rows)

    summary = build_summary(
        experiment_name=args.experiment_name,
        train_csv=train_csv,
        valid_csv=valid_csv,
        output_train_csv=output_train_csv,
        output_valid_csv=output_valid_csv,
        train_rows=train_rows,
        expanded_train_rows=expanded_train_rows,
        valid_rows=valid_rows,
        code_switch_boost_factor=args.code_switch_boost_factor,
        short_clip_boost_factor=args.short_clip_boost_factor,
        short_clip_threshold=args.short_clip_threshold,
    )
    save_strategy_report(reports_dir, summary)

    print(f"Wrote train manifest to {output_train_csv}")
    print(f"Wrote valid manifest to {output_valid_csv}")
    print(f"Wrote strategy summary to {reports_dir / args.experiment_name}")


if __name__ == "__main__":
    main()
