from dataset.repair_metadata_all_block import (
    RepairConfig,
    build_repaired_rows,
    load_reference_fixes,
)


def test_load_reference_fixes_parses_expected_entries(tmp_path) -> None:
    fixes_path = tmp_path / "refrence_fixes.txt"
    fixes_path.write_text(
        "sample_00000010 : corrected text 1\n\nsample_00000011 : corrected text 2\n",
        encoding="utf-8",
    )

    fixes = load_reference_fixes(fixes_path)

    assert fixes == {
        "sample_00000010": "corrected text 1",
        "sample_00000011": "corrected text 2",
    }


def test_build_repaired_rows_deletes_applies_fixes_and_compacts_ids() -> None:
    rows = [
        {"id": "sample_00000007", "text": "old 7", "duration": "1.0"},
        {"id": "sample_00000008", "text": "old 8", "duration": "1.1"},
        {"id": "sample_00000009", "text": "old 9", "duration": "1.2"},
        {"id": "sample_00000010", "text": "old 10", "duration": "1.3"},
        {"id": "sample_00000011", "text": "old 11", "duration": "1.4"},
    ]
    fixes = {"sample_00000010": "fixed 10"}
    config = RepairConfig(delete_start=8, delete_end=9, compact_ids=True)

    repaired_rows, deleted_rows, id_map = build_repaired_rows(rows, fixes, config)

    assert [row["id"] for row in deleted_rows] == ["sample_00000008", "sample_00000009"]
    assert repaired_rows == [
        {"id": "sample_00000007", "text": "old 7", "duration": "1.0"},
        {"id": "sample_00000008", "text": "fixed 10", "duration": "1.3"},
        {"id": "sample_00000009", "text": "old 11", "duration": "1.4"},
    ]
    assert id_map == {
        "sample_00000007": "sample_00000007",
        "sample_00000010": "sample_00000008",
        "sample_00000011": "sample_00000009",
    }
