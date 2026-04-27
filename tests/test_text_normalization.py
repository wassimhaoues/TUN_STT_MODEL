from dataset.text_normalization import NORMALIZATION_VERSION, normalize_transcript


def test_normalization_version_is_defined() -> None:
    assert NORMALIZATION_VERSION == "v1"


def test_normalize_transcript_removes_annotation_tags_and_collapses_whitespace() -> None:
    source = "Aphrodite<\\fr>  كان   صدقني   ربي"
    assert normalize_transcript(source) == "aphrodite كان صدقني ربي"


def test_normalize_transcript_removes_broken_language_suffix_tags() -> None:
    source = "expérience professionnelle/fr>نلقى"
    assert normalize_transcript(source) == "expérience professionnelle نلقي"


def test_normalize_transcript_fixes_script_boundaries_and_invisible_chars() -> None:
    source = "\ufeffالقراية متاعيdes domaines apartبديتها في maisمن قبل"
    expected = "القراية متاعي des domaines apart بديتها في mais من قبل"
    assert normalize_transcript(source) == expected


def test_normalize_transcript_normalizes_arabic_variants() -> None:
    source = "إسمها أم التمر وآنا في المرسى وحتى"
    expected = "اسمها ام التمر وانا في المرسي وحتي"
    assert normalize_transcript(source) == expected


def test_normalize_transcript_preserves_french_apostrophes() -> None:
    source = "c'est l'axe اللي في حياتي"
    expected = "c'est l'axe اللي في حياتي"
    assert normalize_transcript(source) == expected
