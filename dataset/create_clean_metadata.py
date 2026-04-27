from pathlib import Path

import pandas as pd
import soundfile as sf
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "metadata_all.csv"
WAV_DIR = SCRIPT_DIR / "extracted_wavs"

MIN_DURATION = 0.5
MAX_DURATION = 30.0


def main() -> None:
    df = pd.read_csv(CSV_PATH)

    clean_rows: list[dict[str, object]] = []
    removed_rows: list[dict[str, object]] = []

    for row in tqdm(df.itertuples(index=False), total=len(df)):
        sample_id = str(row.id).strip()
        text = str(row.text).strip()
        wav_path = WAV_DIR / f"{sample_id}.wav"

        reason = None
        duration = None

        if not wav_path.exists():
            reason = "missing_wav"
        elif not text or text.lower() == "nan":
            reason = "empty_text"
        else:
            try:
                info = sf.info(str(wav_path))
                duration = round(float(info.duration), 3)

                if duration < MIN_DURATION:
                    reason = "too_short"
                elif duration > MAX_DURATION:
                    reason = "too_long"
            except Exception as exc:
                reason = f"corrupt_wav: {exc}"

        if reason:
            removed_rows.append(
                {
                    "id": sample_id,
                    "text": text,
                    "duration": getattr(row, "duration", None),
                    "reason": reason,
                }
            )
        else:
            clean_rows.append(
                {
                    "id": sample_id,
                    "text": text,
                    "duration": duration,
                }
            )

    clean_df = pd.DataFrame(clean_rows)
    removed_df = pd.DataFrame(removed_rows)

    clean_df.to_csv(SCRIPT_DIR / "metadata_clean.csv", index=False, encoding="utf-8")
    removed_df.to_csv(SCRIPT_DIR / "metadata_removed.csv", index=False, encoding="utf-8")

    print("Original rows:", len(df))
    print("Clean rows:", len(clean_df))
    print("Removed rows:", len(removed_df))
    print("Clean hours:", round(clean_df["duration"].sum() / 3600, 2))
    print("Saved:", SCRIPT_DIR / "metadata_clean.csv")
    print("Saved:", SCRIPT_DIR / "metadata_removed.csv")


if __name__ == "__main__":
    main()
