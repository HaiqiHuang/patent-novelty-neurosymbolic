from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

TRAIN_FILE = RAW_DIR / "patentmatch_train_ultrabalanced.tsv"
TEST_FILE = RAW_DIR / "patentmatch_test_ultrabalanced.tsv"

OUTPUT_TRAIN_FILE = PROCESSED_DIR / "train.csv"
OUTPUT_VALID_FILE = PROCESSED_DIR / "valid.csv"
OUTPUT_TEST_FILE = PROCESSED_DIR / "test.csv"

VALID_RATIO = 0.1
RANDOM_SEED = 42

EXPECTED_COLUMNS = ["patent_application", "prior_art", "novelty"]


def load_and_clean(input_file: Path) -> pd.DataFrame:
    """Load a raw PatentMatch TSV file and convert it into patent novelty format."""
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
            "text": "patent_application",
            "text_b": "prior_art",
            "label": "novelty",
        }
    )

    df = df.dropna(subset=EXPECTED_COLUMNS)

    df["patent_application"] = df["patent_application"].astype(str).str.strip()
    df["prior_art"] = df["prior_art"].astype(str).str.strip()
    df["novelty"] = df["novelty"].astype(int)

    df = df[
        (df["patent_application"] != "")
        & (df["prior_art"] != "")
    ].copy()

    valid_labels = {0, 1}
    invalid_labels = set(df["novelty"].unique()) - valid_labels
    if invalid_labels:
        raise ValueError(f"Invalid novelty labels found: {invalid_labels}")

    return df


def stratified_train_valid_split(
    df: pd.DataFrame,
    valid_ratio: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe into train and valid sets while preserving label distribution."""
    valid_indices = (
        df.groupby("novelty", group_keys=False)
        .sample(frac=valid_ratio, random_state=random_seed)
        .index
    )

    valid_df = df.loc[valid_indices].copy()
    train_df = df.drop(index=valid_indices).copy()

    train_df = train_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    valid_df = valid_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    return train_df, valid_df


def check_columns(split_name: str, split_df: pd.DataFrame) -> None:
    """Check whether a split contains all expected columns."""
    missing_columns = [
        col for col in EXPECTED_COLUMNS
        if col not in split_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{split_name} split is missing columns: {missing_columns}. "
            f"Current columns: {split_df.columns.tolist()}"
        )


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    full_train_df = load_and_clean(TRAIN_FILE)
    test_df = load_and_clean(TEST_FILE)

    train_df, valid_df = stratified_train_valid_split(
        full_train_df,
        valid_ratio=VALID_RATIO,
        random_seed=RANDOM_SEED,
    )

    check_columns("train", train_df)
    check_columns("valid", valid_df)
    check_columns("test", test_df)

    train_df.to_csv(OUTPUT_TRAIN_FILE, index=False)
    valid_df.to_csv(OUTPUT_VALID_FILE, index=False)
    test_df.to_csv(OUTPUT_TEST_FILE, index=False)

    print("Processed files saved:")
    print(f"- {OUTPUT_TRAIN_FILE}")
    print(f"- {OUTPUT_VALID_FILE}")
    print(f"- {OUTPUT_TEST_FILE}")

    print("\nTrain shape:", train_df.shape)
    print("Valid shape:", valid_df.shape)
    print("Test shape:", test_df.shape)

    print("\nTrain novelty distribution:")
    print(train_df["novelty"].value_counts().sort_index())

    print("\nValid novelty distribution:")
    print(valid_df["novelty"].value_counts().sort_index())

    print("\nTest novelty distribution:")
    print(test_df["novelty"].value_counts().sort_index())

    print("\nLabel meaning:")
    print("0 = novel")
    print("1 = not novel")

    print("\nExample processed row:")
    print(train_df.head(1).to_string(index=False))


if __name__ == "__main__":
    main()