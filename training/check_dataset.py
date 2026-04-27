from pathlib import Path

import pandas as pd
import soundfile as sf

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "dataset"
WAV_DIR = DATASET_DIR / "extracted_wavs"


def main() -> None:
    for split in ["train", "valid", "test"]:
        csv_path = DATASET_DIR / f"metadata_{split}.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"Missing: {csv_path}")

        df = pd.read_csv(csv_path)

        print(f"\n=== {split.upper()} ===")
        print("Rows:", len(df))
        print("Columns:", df.columns.tolist())
        print("Total hours:", round(df["duration"].sum() / 3600, 2))
        print("First row:")
        print(df.iloc[0])

        sample_id = df.iloc[0]["id"]
        wav_path = WAV_DIR / f"{sample_id}.wav"

        if not wav_path.exists():
            raise FileNotFoundError(f"Missing WAV: {wav_path}")

        info = sf.info(str(wav_path))
        print("Sample WAV:", wav_path)
        print("Sample rate:", info.samplerate)
        print("Duration:", round(info.duration, 3))

    print("\nDataset check passed.")


if __name__ == "__main__":
    main()
