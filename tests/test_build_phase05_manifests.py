from dataset.build_phase05_manifests import (
    ManifestRow,
    build_summary,
    expand_training_rows,
    has_latin,
)


def make_row(sample_id: str, text: str, duration: float) -> ManifestRow:
    return ManifestRow(
        id=sample_id,
        text=text,
        duration=duration,
        text_raw=text,
        normalization_changed="False",
        normalization_version="v1",
    )


def test_expand_training_rows_boosts_code_switch_and_short_clips() -> None:
    rows = [
        make_row("sample_1", "bonjour", 5.0),
        make_row("sample_2", "مرحبا", 2.0),
        make_row("sample_3", "salut مرحبا", 2.5),
        make_row("sample_4", "اهلا", 5.0),
    ]

    expanded = expand_training_rows(
        rows=rows,
        code_switch_boost_factor=3,
        short_clip_boost_factor=2,
        short_clip_threshold=3.0,
    )

    expanded_ids = [row.id for row in expanded]
    assert expanded_ids.count("sample_1") == 3
    assert expanded_ids.count("sample_2") == 2
    assert expanded_ids.count("sample_3") == 3
    assert expanded_ids.count("sample_4") == 1


def test_build_summary_counts_target_rows(tmp_path) -> None:
    train_rows = [
        make_row("sample_1", "bonjour", 5.0),
        make_row("sample_2", "مرحبا", 2.0),
        make_row("sample_3", "salut مرحبا", 2.5),
    ]
    expanded_rows = expand_training_rows(
        rows=train_rows,
        code_switch_boost_factor=2,
        short_clip_boost_factor=2,
        short_clip_threshold=3.0,
    )
    valid_rows = [make_row("sample_4", "اهلا", 4.0)]

    summary = build_summary(
        experiment_name="phase05-test",
        train_csv=tmp_path / "train.csv",
        valid_csv=tmp_path / "valid.csv",
        output_train_csv=tmp_path / "out-train.csv",
        output_valid_csv=tmp_path / "out-valid.csv",
        train_rows=train_rows,
        expanded_train_rows=expanded_rows,
        valid_rows=valid_rows,
        code_switch_boost_factor=2,
        short_clip_boost_factor=2,
        short_clip_threshold=3.0,
    )

    assert has_latin(train_rows[0].text)
    assert summary.train_rows_in == 3
    assert summary.train_rows_out == 6
    assert summary.code_switched_train_rows == 2
    assert summary.short_train_rows == 2
    assert summary.both_code_switched_and_short_rows == 1
