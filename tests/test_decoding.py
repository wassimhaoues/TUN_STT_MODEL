from training.decoding import (
    PHASE05_SAFE_DECODE_PRESET,
    build_generate_kwargs,
    resolve_decoding_config,
)


def test_phase05_safe_decode_preset_enables_repeat_guards() -> None:
    config = resolve_decoding_config(
        preset=PHASE05_SAFE_DECODE_PRESET,
        language="arabic",
        task="transcribe",
        generation_max_length=225,
    )

    assert config.generation_num_beams == 3
    assert config.generation_no_repeat_ngram_size == 3
    assert config.generation_repetition_penalty == 1.1


def test_generate_kwargs_drop_no_repeat_when_disabled() -> None:
    config = resolve_decoding_config(
        preset="standard",
        language="arabic",
        task="transcribe",
        generation_max_length=225,
    )

    kwargs = build_generate_kwargs(config)

    assert kwargs["num_beams"] == 1
    assert "no_repeat_ngram_size" not in kwargs
