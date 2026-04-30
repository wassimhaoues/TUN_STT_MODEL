from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "dataset"
DEFAULT_METADATA_ALL = DATASET_DIR / "metadata_all.csv"
DEFAULT_FIXES_PATH = DATASET_DIR / "refrence_fixes.txt"
DEFAULT_AUDIO_DIR = DATASET_DIR / "extracted_wavs"
DEFAULT_BACKUP_ROOT = DATASET_DIR / "repairs" / "phase05_alignment_repair"
SAMPLE_ID_PATTERN = re.compile(r"^sample_(\d{8})$")


@dataclass(frozen=True)
class RepairConfig:
    delete_start: int
    delete_end: int
    compact_ids: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repair metadata_all.csv using a reference-fix file, remove a corrupted "
            "audio block, and optionally compact later sample ids."
        )
    )
    parser.add_argument(
        "--metadata-all",
        default=str(DEFAULT_METADATA_ALL),
        help="Path to metadata_all.csv.",
    )
    parser.add_argument(
        "--fixes-path",
        default=str(DEFAULT_FIXES_PATH),
        help="Path to the text file that contains corrected references.",
    )
    parser.add_argument(
        "--audio-dir",
        default=str(DEFAULT_AUDIO_DIR),
        help="Directory that contains extracted wav files.",
    )
    parser.add_argument(
        "--backup-root",
        default=str(DEFAULT_BACKUP_ROOT),
        help="Directory where backups and removed audio will be stored.",
    )
    parser.add_argument(
        "--delete-start",
        type=int,
        default=13107,
        help="First sample number in the corrupted block to remove.",
    )
    parser.add_argument(
        "--delete-end",
        type=int,
        default=13115,
        help="Last sample number in the corrupted block to remove.",
    )
    parser.add_argument(
        "--compact-ids",
        action="store_true",
        help=(
            "Shift later sample ids down so numbering stays contiguous. "
            "This will rename all later wav files too."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned changes without modifying files.",
    )
    return parser.parse_args()


def sample_id(number: int) -> str:
    return f"sample_{number:08d}"


def sample_number(sample_name: str) -> int:
    match = SAMPLE_ID_PATTERN.fullmatch(sample_name)
    if not match:
        raise ValueError(f"Invalid sample id: {sample_name}")
    return int(match.group(1))


def load_metadata_all(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    if fieldnames != ["id", "text", "duration"]:
        raise ValueError(f"Unexpected metadata_all header: {fieldnames}")
    return fieldnames, rows


def load_reference_fixes(path: Path) -> dict[str, str]:
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return {}

    fixes: dict[str, str] = {}
    chunks = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]
    for chunk in chunks:
        first_colon = chunk.find(":")
        if first_colon == -1:
            raise ValueError(f"Invalid fixes entry, missing ':' in: {chunk}")
        key = chunk[:first_colon].strip()
        value = chunk[first_colon + 1 :].strip()
        if key in fixes:
            raise ValueError(f"Duplicate fixes entry for {key}")
        sample_number(key)
        fixes[key] = value
    return fixes


def build_repaired_rows(
    rows: list[dict[str, str]],
    fixes: dict[str, str],
    config: RepairConfig,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, str]]:
    deleted_rows: list[dict[str, str]] = []
    repaired_rows: list[dict[str, str]] = []
    id_map: dict[str, str] = {}
    removed_count = config.delete_end - config.delete_start + 1

    for row in rows:
        old_id = row["id"]
        number = sample_number(old_id)

        if config.delete_start <= number <= config.delete_end:
            deleted_rows.append(dict(row))
            continue

        updated_row = dict(row)
        if old_id in fixes:
            updated_row["text"] = fixes[old_id]

        new_number = number
        if config.compact_ids and number > config.delete_end:
            new_number = number - removed_count

        new_id = sample_id(new_number)
        updated_row["id"] = new_id
        repaired_rows.append(updated_row)
        id_map[old_id] = new_id

    return repaired_rows, deleted_rows, id_map


def write_metadata_all(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def ensure_files_exist(audio_dir: Path, ids: list[str]) -> None:
    missing = [
        sample_name for sample_name in ids if not (audio_dir / f"{sample_name}.wav").exists()
    ]
    if missing:
        raise FileNotFoundError("Missing wav files for sample ids: " + ", ".join(missing[:10]))


def move_deleted_audio(
    audio_dir: Path,
    deleted_ids: list[str],
    removed_audio_dir: Path,
) -> list[dict[str, str]]:
    removed_audio_dir.mkdir(parents=True, exist_ok=True)
    moved: list[dict[str, str]] = []
    for sample_name in deleted_ids:
        source = audio_dir / f"{sample_name}.wav"
        target = removed_audio_dir / f"{sample_name}.wav"
        shutil.move(str(source), str(target))
        moved.append({"old_path": str(source), "new_path": str(target)})
    return moved


def rename_audio_files(
    audio_dir: Path,
    id_map: dict[str, str],
    untouched_ids: set[str],
) -> list[dict[str, str]]:
    renamed: list[dict[str, str]] = []
    temp_moves: list[tuple[Path, Path, Path]] = []

    for old_id, new_id in id_map.items():
        if old_id == new_id or old_id in untouched_ids:
            continue
        source = audio_dir / f"{old_id}.wav"
        temp = audio_dir / f"{old_id}.wav.repair_tmp"
        final = audio_dir / f"{new_id}.wav"
        shutil.move(str(source), str(temp))
        temp_moves.append((source, temp, final))

    for source, temp, final in temp_moves:
        shutil.move(str(temp), str(final))
        renamed.append({"old_path": str(source), "new_path": str(final)})

    return renamed


def backup_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def write_removed_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "text", "duration"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_operation_log(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    metadata_all_path = Path(args.metadata_all).resolve()
    fixes_path = Path(args.fixes_path).resolve()
    audio_dir = Path(args.audio_dir).resolve()
    backup_root = Path(args.backup_root).resolve()
    config = RepairConfig(
        delete_start=args.delete_start,
        delete_end=args.delete_end,
        compact_ids=args.compact_ids,
    )

    if config.delete_end < config.delete_start:
        raise ValueError("--delete-end must be greater than or equal to --delete-start")

    fieldnames, rows = load_metadata_all(metadata_all_path)
    fixes = load_reference_fixes(fixes_path)
    repaired_rows, deleted_rows, id_map = build_repaired_rows(rows, fixes, config)

    deleted_ids = [row["id"] for row in deleted_rows]
    ensure_files_exist(audio_dir, deleted_ids)
    if config.compact_ids:
        ensure_files_exist(
            audio_dir,
            [old_id for old_id, new_id in id_map.items() if old_id != new_id],
        )

    removed_count = len(deleted_rows)
    fixed_count = sum(1 for row_id in fixes if row_id in {row["id"] for row in rows})
    preview = {
        "metadata_all": str(metadata_all_path),
        "fixes_path": str(fixes_path),
        "audio_dir": str(audio_dir),
        "delete_range": [sample_id(config.delete_start), sample_id(config.delete_end)],
        "deleted_rows": removed_count,
        "fixed_rows": fixed_count,
        "compacted_ids": config.compact_ids,
        "final_row_count": len(repaired_rows),
    }

    if args.dry_run:
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return

    backup_root.mkdir(parents=True, exist_ok=True)
    backup_file(metadata_all_path, backup_root / "backups" / "metadata_all.csv")
    backup_file(fixes_path, backup_root / "backups" / fixes_path.name)

    removed_rows_path = backup_root / "removed_rows.csv"
    removed_audio_dir = backup_root / "removed_audio"
    operation_log_path = backup_root / "operation_log.json"

    moved_audio = move_deleted_audio(audio_dir, deleted_ids, removed_audio_dir)
    renamed_audio = rename_audio_files(audio_dir, id_map, untouched_ids=set(deleted_ids))
    write_metadata_all(metadata_all_path, fieldnames, repaired_rows)
    write_removed_rows(removed_rows_path, deleted_rows)
    write_operation_log(
        operation_log_path,
        {
            **preview,
            "deleted_ids": deleted_ids,
            "moved_audio": moved_audio,
            "renamed_audio_count": len(renamed_audio),
            "renamed_audio_preview": renamed_audio[:20],
        },
    )

    print(f"Updated {metadata_all_path}")
    print(
        f"Removed {removed_count} corrupted rows and moved their wav files to {removed_audio_dir}"
    )
    print(f"Applied {fixed_count} reference fixes from {fixes_path.name}")
    if config.compact_ids:
        print(f"Compacted later sample ids and renamed {len(renamed_audio)} wav files")
    print(f"Backups and logs saved under {backup_root}")


if __name__ == "__main__":
    main()
