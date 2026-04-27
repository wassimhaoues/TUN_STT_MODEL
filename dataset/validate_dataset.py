import re
from pathlib import Path

import pandas as pd
import soundfile as sf
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "metadata_all.csv"
WAV_DIR = SCRIPT_DIR / "extracted_wavs"

MIN_DURATION = 0.5
MAX_DURATION = 30.0

latin_pattern = re.compile(r"[A-Za-z]")


def main() -> None:
    df = pd.read_csv(CSV_PATH)

    required_cols = {"id", "text", "duration"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    report = {
        "total_rows": len(df),
        "missing_wav": 0,
        "corrupt_wav": 0,
        "empty_text": 0,
        "too_short": 0,
        "too_long": 0,
        "code_switched_latin": 0,
        "duration_mismatch": 0,
    }
    bad_rows: list[dict[str, object]] = []
    durations: list[float] = []

    for idx, row in enumerate(tqdm(df.itertuples(index=False), total=len(df))):
        sample_id = str(row.id).strip()
        text = str(row.text).strip()
        csv_duration = float(row.duration)
        wav_path = WAV_DIR / f"{sample_id}.wav"

        problems: list[str] = []

        if not text or text.lower() == "nan":
            report["empty_text"] += 1
            problems.append("empty_text")

        if latin_pattern.search(text):
            report["code_switched_latin"] += 1

        if not wav_path.exists():
            report["missing_wav"] += 1
            problems.append("missing_wav")
        else:
            try:
                info = sf.info(str(wav_path))
                real_duration = info.duration
                durations.append(real_duration)

                if real_duration < MIN_DURATION:
                    report["too_short"] += 1
                    problems.append("too_short")

                if real_duration > MAX_DURATION:
                    report["too_long"] += 1
                    problems.append("too_long")

                if abs(real_duration - csv_duration) > 0.2:
                    report["duration_mismatch"] += 1
                    problems.append(f"duration_mismatch_real={real_duration:.3f}")
            except Exception as exc:
                report["corrupt_wav"] += 1
                problems.append(f"corrupt_wav: {exc}")

        if problems:
            bad_rows.append(
                {
                    "row_index": idx,
                    "id": sample_id,
                    "text": text,
                    "duration": csv_duration,
                    "problems": " | ".join(problems),
                }
            )

    print("\n====================")
    print("DATASET REPORT")
    print("====================")

    for key, value in report.items():
        print(f"{key}: {value}")

    if durations:
        duration_series = pd.Series(durations)
        print("\nDuration stats:")
        print(duration_series.describe())
        print("\nTotal hours:", round(duration_series.sum() / 3600, 2))

    bad_df = pd.DataFrame(bad_rows)
    bad_df.to_csv(SCRIPT_DIR / "bad_rows_report.csv", index=False, encoding="utf-8")
    print("\nBad rows saved to:", SCRIPT_DIR / "bad_rows_report.csv")


if __name__ == "__main__":
    main()
