from dataset.build_phase05_manifests import (
    AuditDecision,
    ManifestRow,
    build_summary,
    expand_training_rows,
    has_latin,
    load_audit_decisions,
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
        audit_decisions={},
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
        audit_decisions={},
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
        audit_decisions={},
        audit_csv=None,
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


def test_load_audit_decisions_parses_yes_and_fix_text(tmp_path) -> None:
    audit_csv = tmp_path / "audit.csv"
    audit_csv.write_text(
        (
            "id,transcript_action,audio_action,keep_for_phase05_boost,corrected_text\n"
            "sample_1,fix_text,keep_raw,yes,bonjour corrige\n"
        ),
        encoding="utf-8",
    )

    decisions = load_audit_decisions(audit_csv)

    assert decisions["sample_1"] == AuditDecision(
        sample_id="sample_1",
        transcript_action="fix_text",
        audio_action="keep_raw",
        keep_for_phase05_boost=True,
        corrected_text="bonjour corrige",
    )


def test_expand_training_rows_respects_audit_exclude_and_boost_gate() -> None:
    rows = [
        make_row("sample_1", "bonjour", 5.0),
        make_row("sample_2", "salut مرحبا", 2.5),
        make_row("sample_3", "اهلا", 2.0),
    ]
    decisions = {
        "sample_1": AuditDecision(
            sample_id="sample_1",
            transcript_action="fix_text",
            audio_action="keep_raw",
            keep_for_phase05_boost=False,
            corrected_text="bonjour corrige",
        ),
        "sample_2": AuditDecision(
            sample_id="sample_2",
            transcript_action="keep",
            audio_action="exclude",
            keep_for_phase05_boost=True,
            corrected_text="",
        ),
        "sample_3": AuditDecision(
            sample_id="sample_3",
            transcript_action="keep",
            audio_action="keep_raw",
            keep_for_phase05_boost=True,
            corrected_text="",
        ),
    }

    expanded = expand_training_rows(
        rows=rows,
        audit_decisions=decisions,
        code_switch_boost_factor=3,
        short_clip_boost_factor=2,
        short_clip_threshold=3.0,
    )

    assert [row.id for row in expanded] == ["sample_1", "sample_3", "sample_3"]
    assert expanded[0].text == "bonjour corrige"


def test_build_summary_counts_audit_effects(tmp_path) -> None:
    train_rows = [
        make_row("sample_1", "bonjour", 5.0),
        make_row("sample_2", "salut مرحبا", 2.5),
        make_row("sample_3", "اهلا", 2.0),
    ]
    decisions = {
        "sample_1": AuditDecision(
            sample_id="sample_1",
            transcript_action="fix_text",
            audio_action="gain_normalize",
            keep_for_phase05_boost=True,
            corrected_text="bonjour corrige",
        ),
        "sample_2": AuditDecision(
            sample_id="sample_2",
            transcript_action="exclude",
            audio_action="keep_raw",
            keep_for_phase05_boost=False,
            corrected_text="",
        ),
    }
    expanded_rows = expand_training_rows(
        rows=train_rows,
        audit_decisions=decisions,
        code_switch_boost_factor=2,
        short_clip_boost_factor=2,
        short_clip_threshold=3.0,
    )
    summary = build_summary(
        experiment_name="phase05-audit",
        train_csv=tmp_path / "train.csv",
        valid_csv=tmp_path / "valid.csv",
        output_train_csv=tmp_path / "out-train.csv",
        output_valid_csv=tmp_path / "out-valid.csv",
        train_rows=train_rows,
        expanded_train_rows=expanded_rows,
        valid_rows=[make_row("sample_4", "اهلا", 4.0)],
        audit_decisions=decisions,
        audit_csv=tmp_path / "audit.csv",
        code_switch_boost_factor=2,
        short_clip_boost_factor=2,
        short_clip_threshold=3.0,
    )

    assert summary.excluded_train_rows == 1
    assert summary.corrected_train_rows == 1
    assert summary.boost_approved_rows == 1
    assert summary.gain_normalize_flag_rows == 1
