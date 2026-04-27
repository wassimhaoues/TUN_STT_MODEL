from pathlib import Path
import pandas as pd
import soundfile as sf
from tqdm import tqdm

CSV_PATH = Path("metadata_all.csv")
WAV_DIR = Path("extracted_wavs")

MIN_DURATION = 0.5
MAX_DURATION = 30.0

df = pd.read_csv(CSV_PATH)

clean_rows = []
removed_rows = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    sample_id = str(row["id"]).strip()
    text = str(row["text"]).strip()
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

        except Exception as e:
            reason = f"corrupt_wav: {e}"

    if reason:
        removed_rows.append({
            "id": sample_id,
            "text": text,
            "duration": row.get("duration", None),
            "reason": reason,
        })
    else:
        clean_rows.append({
            "id": sample_id,
            "text": text,
            "duration": duration,
        })

clean_df = pd.DataFrame(clean_rows)
removed_df = pd.DataFrame(removed_rows)

clean_df.to_csv("metadata_clean.csv", index=False, encoding="utf-8")
removed_df.to_csv("metadata_removed.csv", index=False, encoding="utf-8")

print("Original rows:", len(df))
print("Clean rows:", len(clean_df))
print("Removed rows:", len(removed_df))
print("Clean hours:", round(clean_df["duration"].sum() / 3600, 2))
print("Saved: metadata_clean.csv")
print("Saved: metadata_removed.csv")