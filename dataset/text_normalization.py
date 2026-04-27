from __future__ import annotations

import re
import unicodedata

NORMALIZATION_VERSION = "v1"

ARABIC_TO_LATIN_BOUNDARY_PATTERN = re.compile(r"([\u0600-\u06FF])([A-Za-zÀ-ÖØ-öø-ÿ])")
LATIN_TO_ARABIC_BOUNDARY_PATTERN = re.compile(r"([A-Za-zÀ-ÖØ-öø-ÿ])([\u0600-\u06FF])")
BROKEN_LANGUAGE_TAG_PATTERN = re.compile(
    r"(?:<\s*[\\/](?:fr|en)\s*>|[\\/](?:fr|en)\s*>)",
    flags=re.IGNORECASE,
)
GENERIC_ANGLE_TAG_PATTERN = re.compile(r"<[^>]+>")
INVISIBLE_CHARACTER_PATTERN = re.compile(r"[\u200b-\u200f\u2060\ufeff]")
WHITESPACE_PATTERN = re.compile(r"\s+")
ARABIC_CHAR_TRANSLATION = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ى": "ي",
    }
)


def normalize_transcript(text: str) -> str:
    normalized_text = unicodedata.normalize("NFKC", str(text))
    normalized_text = INVISIBLE_CHARACTER_PATTERN.sub("", normalized_text)
    normalized_text = BROKEN_LANGUAGE_TAG_PATTERN.sub(" ", normalized_text)
    normalized_text = GENERIC_ANGLE_TAG_PATTERN.sub(" ", normalized_text)
    normalized_text = normalized_text.lower()
    normalized_text = normalized_text.translate(ARABIC_CHAR_TRANSLATION)
    normalized_text = ARABIC_TO_LATIN_BOUNDARY_PATTERN.sub(r"\1 \2", normalized_text)
    normalized_text = LATIN_TO_ARABIC_BOUNDARY_PATTERN.sub(r"\1 \2", normalized_text)
    normalized_text = WHITESPACE_PATTERN.sub(" ", normalized_text)
    return normalized_text.strip()
