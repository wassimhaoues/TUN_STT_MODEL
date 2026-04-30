from training.transcribe_one_audio import compute_cer, compute_wer, levenshtein_distance


def test_levenshtein_distance_handles_words() -> None:
    assert levenshtein_distance(["a", "b"], ["a", "c"]) == 1


def test_compute_wer_and_cer() -> None:
    assert compute_wer("مرحبا بيك", "مرحبا") == 0.5
    assert compute_cer("abc", "adc") == 0.3333333333333333
