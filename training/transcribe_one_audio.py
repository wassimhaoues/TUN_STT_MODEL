from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataset.text_normalization import NORMALIZATION_VERSION, normalize_transcript  # noqa: E402
from training.baseline_test import (  # noqa: E402
    DEFAULT_LANGUAGE,
    DEFAULT_TASK,
    MODEL_NAME,
    format_metric,
    get_device_config,
    load_audio_for_asr,
)
from training.decoding import (  # noqa: E402
    DECODING_PRESETS,
    DEFAULT_DECODING_PRESET,
    apply_decoding_config,
    build_generate_kwargs,
    resolve_decoding_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe a single audio file with the baseline model or a local checkpoint."
    )
    parser.add_argument(
        "--audio-path",
        required=True,
        help="Path to the WAV audio file to transcribe.",
    )
    parser.add_argument(
        "--model-path",
        default=MODEL_NAME,
        help=(
            "Checkpoint directory or model path to use. "
            f"Defaults to the baseline model `{MODEL_NAME}`."
        ),
    )
    parser.add_argument(
        "--reference-text",
        help="Optional reference transcript. When provided, WER and CER are reported.",
    )
    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help="Whisper generation language.",
    )
    parser.add_argument(
        "--task",
        default=DEFAULT_TASK,
        help="Whisper generation task.",
    )
    parser.add_argument(
        "--generation-max-length",
        type=int,
        default=225,
        help="Generation max length.",
    )
    parser.add_argument(
        "--decoding-preset",
        choices=list(DECODING_PRESETS),
        default=DEFAULT_DECODING_PRESET,
        help="Tracked decoding preset for single-audio transcription.",
    )
    parser.add_argument("--generation-num-beams", type=int, help="Optional beam-count override.")
    parser.add_argument(
        "--generation-length-penalty",
        type=float,
        help="Optional length-penalty override.",
    )
    parser.add_argument(
        "--generation-no-repeat-ngram-size",
        type=int,
        help="Optional no-repeat ngram override. Use 0 to disable.",
    )
    parser.add_argument(
        "--generation-repetition-penalty",
        type=float,
        help="Optional repetition-penalty override.",
    )
    return parser.parse_args()


def levenshtein_distance(left: list[str], right: list[str]) -> int:
    if not left:
        return len(right)
    if not right:
        return len(left)

    distances = list(range(len(right) + 1))
    for left_index, left_token in enumerate(left, start=1):
        previous_diagonal = distances[0]
        distances[0] = left_index
        for right_index, right_token in enumerate(right, start=1):
            current = distances[right_index]
            distances[right_index] = min(
                distances[right_index] + 1,
                distances[right_index - 1] + 1,
                previous_diagonal + int(left_token != right_token),
            )
            previous_diagonal = current
    return distances[-1]


def compute_wer(reference: str, prediction: str) -> float:
    reference_words = reference.split()
    prediction_words = prediction.split()
    edits = levenshtein_distance(reference_words, prediction_words)
    return edits / max(1, len(reference_words))


def compute_cer(reference: str, prediction: str) -> float:
    edits = levenshtein_distance(list(reference), list(prediction))
    return edits / max(1, len(reference))


def main() -> None:
    args = parse_args()
    audio_path = Path(args.audio_path).resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    device_index, device_name = get_device_config()
    processor = WhisperProcessor.from_pretrained(args.model_path)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_path)
    model_device = "cpu" if device_index < 0 else device_name
    model.to(model_device)

    decoding_config = resolve_decoding_config(
        preset=args.decoding_preset,
        language=args.language,
        task=args.task,
        generation_max_length=args.generation_max_length,
        generation_num_beams=args.generation_num_beams,
        generation_length_penalty=args.generation_length_penalty,
        generation_no_repeat_ngram_size=args.generation_no_repeat_ngram_size,
        generation_repetition_penalty=args.generation_repetition_penalty,
    )
    apply_decoding_config(model, decoding_config)

    import torch

    audio_input = load_audio_for_asr(audio_path)
    inputs = processor(
        audio_input["raw"],
        sampling_rate=audio_input["sampling_rate"],
        return_tensors="pt",
    )
    input_features = inputs.input_features.to(model_device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_features,
            **build_generate_kwargs(decoding_config),
        )

    prediction = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    normalized_prediction = normalize_transcript(prediction)

    print(f"Audio path: {audio_path}")
    print(f"Model path: {args.model_path}")
    print(f"Device: {device_name}")
    print(f"Decoding preset: {args.decoding_preset}")
    print(f"Normalization version: {NORMALIZATION_VERSION}")
    print("")
    print("Prediction:")
    print(prediction)
    print("")
    print("Normalized prediction:")
    print(normalized_prediction)

    if args.reference_text:
        normalized_reference = normalize_transcript(args.reference_text.strip())
        wer = compute_wer(normalized_reference, normalized_prediction)
        cer = compute_cer(normalized_reference, normalized_prediction)
        print("")
        print("Reference:")
        print(args.reference_text.strip())
        print("")
        print("Normalized reference:")
        print(normalized_reference)
        print("")
        print(f"WER: {format_metric(wer)}")
        print(f"CER: {format_metric(cer)}")


if __name__ == "__main__":
    main()
