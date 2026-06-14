from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

TRAIN_FILE = RAW_DIR / "patentmatch_train_ultrabalanced.tsv"
TEST_FILE = RAW_DIR / "patentmatch_test_ultrabalanced.tsv"

OUTPUT_TRAIN_FILE = PROCESSED_DIR / "train.csv"
OUTPUT_TEST_FILE = PROCESSED_DIR / "test.csv"


def load_and_clean(input_file: Path) -> pd.DataFrame:
    """Load a raw PatentMatch TSV file and convert it into sentence-pair format."""
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_csv(input_file, sep="\t")

    required_columns = ["text", "text_b", "label"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns in {input_file}: {missing_columns}")

    df = df[required_columns].copy()

    df = df.rename(
        columns={
            "text": "text_a",
            "text_b": "text_b",
            "label": "label",
        }
    )

    df["text_a"] = df["text_a"].astype(str).str.strip()
    df["text_b"] = df["text_b"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)

    df = df.dropna(subset=["text_a", "text_b", "label"])
    df = df[(df["text_a"] != "") & (df["text_b"] != "")]

    return df


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_and_clean(TRAIN_FILE)
    test_df = load_and_clean(TEST_FILE)

    train_df.to_csv(OUTPUT_TRAIN_FILE, index=False)
    test_df.to_csv(OUTPUT_TEST_FILE, index=False)

    print("Processed files saved:")
    print(f"- {OUTPUT_TRAIN_FILE}")
    print(f"- {OUTPUT_TEST_FILE}")

    print("\nTrain shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    print("\nTrain label distribution:")
    print(train_df["label"].value_counts())

    print("\nTest label distribution:")
    print(test_df["label"].value_counts())

    print("\nExample processed row:")
    print(train_df.head(1).to_string(index=False))


if __name__ == "__main__":
    main()