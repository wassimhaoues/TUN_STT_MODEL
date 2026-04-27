from __future__ import annotations

import csv
import py_compile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
PYTHON_FILES = sorted((ROOT_DIR / "dataset").glob("*.py")) + sorted(
    (ROOT_DIR / "training").glob("*.py")
)
METADATA_FILES = [
    ROOT_DIR / "dataset" / "metadata_clean.csv",
    ROOT_DIR / "dataset" / "metadata_train.csv",
    ROOT_DIR / "dataset" / "metadata_valid.csv",
    ROOT_DIR / "dataset" / "metadata_test.csv",
]
EXPECTED_COLUMNS = {"id", "text", "duration"}


def test_python_scripts_compile() -> None:
    assert PYTHON_FILES, "Expected project Python scripts to exist."

    for path in PYTHON_FILES:
        py_compile.compile(str(path), doraise=True)


def test_metadata_files_have_expected_columns() -> None:
    for path in METADATA_FILES:
        assert path.exists(), f"Missing metadata file: {path}"

        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            assert reader.fieldnames is not None, f"Missing CSV header in {path}"
            assert EXPECTED_COLUMNS.issubset(reader.fieldnames), (
                f"{path} is missing one of the required columns: {sorted(EXPECTED_COLUMNS)}"
            )
