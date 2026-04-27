from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent


def main() -> None:
    df = pd.read_csv(SCRIPT_DIR / "metadata_clean.csv")

    # Keep a balanced ratio of code-switched rows across all splits.
    df["has_latin"] = df["text"].astype(str).str.contains(r"[A-Za-z]", regex=True)

    train_df, temp_df = train_test_split(
        df,
        test_size=0.10,
        random_state=42,
        shuffle=True,
        stratify=df["has_latin"],
    )

    valid_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        shuffle=True,
        stratify=temp_df["has_latin"],
    )

    for split_df, name in [
        (train_df, "metadata_train.csv"),
        (valid_df, "metadata_valid.csv"),
        (test_df, "metadata_test.csv"),
    ]:
        split_df = split_df.drop(columns=["has_latin"])
        split_df.to_csv(SCRIPT_DIR / name, index=False, encoding="utf-8")

    print("Train rows:", len(train_df), "hours:", round(train_df["duration"].sum() / 3600, 2))
    print("Valid rows:", len(valid_df), "hours:", round(valid_df["duration"].sum() / 3600, 2))
    print("Test rows:", len(test_df), "hours:", round(test_df["duration"].sum() / 3600, 2))

    print("\nCode-switch ratio:")
    print("Train:", round(train_df["has_latin"].mean() * 100, 2), "%")
    print("Valid:", round(valid_df["has_latin"].mean() * 100, 2), "%")
    print("Test:", round(test_df["has_latin"].mean() * 100, 2), "%")


if __name__ == "__main__":
    main()
