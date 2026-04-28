from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

DEFAULT_DECODING_PRESET = "standard"
PHASE05_SAFE_DECODE_PRESET = "phase05_safe_decode_v1"
DECODING_PRESETS = (
    DEFAULT_DECODING_PRESET,
    PHASE05_SAFE_DECODE_PRESET,
)


@dataclass(frozen=True)
class DecodingConfig:
    preset: str
    language: str
    task: str
    generation_max_length: int
    generation_num_beams: int
    generation_length_penalty: float
    generation_no_repeat_ngram_size: int
    generation_repetition_penalty: float


def resolve_decoding_config(
    preset: str,
    language: str,
    task: str,
    generation_max_length: int,
    generation_num_beams: int | None = None,
    generation_length_penalty: float | None = None,
    generation_no_repeat_ngram_size: int | None = None,
    generation_repetition_penalty: float | None = None,
) -> DecodingConfig:
    if preset not in DECODING_PRESETS:
        raise ValueError(f"Unknown decoding preset: {preset}")

    if preset == PHASE05_SAFE_DECODE_PRESET:
        base_num_beams = 3
        base_length_penalty = 1.0
        base_no_repeat_ngram_size = 3
        base_repetition_penalty = 1.10
    else:
        base_num_beams = 1
        base_length_penalty = 1.0
        base_no_repeat_ngram_size = 0
        base_repetition_penalty = 1.0

    config = DecodingConfig(
        preset=preset,
        language=language,
        task=task,
        generation_max_length=generation_max_length,
        generation_num_beams=generation_num_beams or base_num_beams,
        generation_length_penalty=(
            generation_length_penalty
            if generation_length_penalty is not None
            else base_length_penalty
        ),
        generation_no_repeat_ngram_size=(
            generation_no_repeat_ngram_size
            if generation_no_repeat_ngram_size is not None
            else base_no_repeat_ngram_size
        ),
        generation_repetition_penalty=(
            generation_repetition_penalty
            if generation_repetition_penalty is not None
            else base_repetition_penalty
        ),
    )
    validate_decoding_config(config)
    return config


def validate_decoding_config(config: DecodingConfig) -> None:
    if config.generation_max_length <= 0:
        raise ValueError("generation_max_length must be greater than 0.")
    if config.generation_num_beams <= 0:
        raise ValueError("generation_num_beams must be greater than 0.")
    if config.generation_no_repeat_ngram_size < 0:
        raise ValueError("generation_no_repeat_ngram_size must be 0 or greater.")
    if config.generation_length_penalty <= 0:
        raise ValueError("generation_length_penalty must be greater than 0.")
    if config.generation_repetition_penalty <= 0:
        raise ValueError("generation_repetition_penalty must be greater than 0.")


def apply_decoding_config(model: Any, config: DecodingConfig) -> None:
    model.generation_config.language = config.language
    model.generation_config.task = config.task
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.generation_config.max_length = config.generation_max_length
    model.generation_config.num_beams = config.generation_num_beams
    model.generation_config.length_penalty = config.generation_length_penalty
    model.generation_config.no_repeat_ngram_size = config.generation_no_repeat_ngram_size
    model.generation_config.repetition_penalty = config.generation_repetition_penalty


def build_generate_kwargs(config: DecodingConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "language": config.language,
        "task": config.task,
        "max_length": config.generation_max_length,
        "num_beams": config.generation_num_beams,
        "length_penalty": config.generation_length_penalty,
        "repetition_penalty": config.generation_repetition_penalty,
    }
    if config.generation_no_repeat_ngram_size > 0:
        kwargs["no_repeat_ngram_size"] = config.generation_no_repeat_ngram_size
    return kwargs


def decoding_config_to_dict(config: DecodingConfig) -> dict[str, Any]:
    return asdict(config)
