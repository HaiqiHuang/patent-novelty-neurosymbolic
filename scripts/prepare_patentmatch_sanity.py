from pathlib import Path

import numpy as np
import pandas as pd


INPUT_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/sanity_test")

TRAIN_FILE = INPUT_DIR / "train.csv"
VALID_FILE = INPUT_DIR / "valid.csv"

REQUIRED_COLUMNS = ["patent_application", "prior_art", "novelty"]


def load_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"{path} is missing columns: {missing_columns}")

    df["patent_application"] = df["patent_application"].fillna("").astype(str)
    df["prior_art"] = df["prior_art"].fillna("").astype(str)
    df["novelty"] = df["novelty"].astype(int)

    return df


def save_baseline(train_df: pd.DataFrame, valid_df: pd.DataFrame, name: str) -> None:
    baseline_dir = OUTPUT_DIR / name
    baseline_dir.mkdir(parents=True, exist_ok=True)

    train_path = baseline_dir / "train.csv"
    valid_path = baseline_dir / "valid.csv"

    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)

    print(f"Saved {name}")
    print(f"  train: {train_path}")
    print(f"  valid: {valid_path}")
    print("  train labels:")
    print(train_df["novelty"].value_counts())
    print("  valid labels:")
    print(valid_df["novelty"].value_counts())
    print()


def shuffle_prior_art(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    shuffled = df.copy()
    rng = np.random.default_rng(seed)

    indices = np.arange(len(shuffled))
    permuted = rng.permutation(indices)

    # Avoid fixed points where possible.
    if np.any(permuted == indices):
        permuted = np.roll(permuted, 1)

    shuffled["prior_art"] = shuffled["prior_art"].iloc[permuted].to_numpy()
    return shuffled


def main() -> None:
    train = load_split(TRAIN_FILE)
    valid = load_split(VALID_FILE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Claim-only baseline
    claim_only_train = train.copy()
    claim_only_valid = valid.copy()
    claim_only_train["prior_art"] = " "
    claim_only_valid["prior_art"] = " "
    save_baseline(claim_only_train, claim_only_valid, "claim_only")

    # 2. Prior-art-only baseline
    prior_art_only_train = train.copy()
    prior_art_only_valid = valid.copy()
    prior_art_only_train["patent_application"] = " "
    prior_art_only_valid["patent_application"] = " "
    save_baseline(prior_art_only_train, prior_art_only_valid, "prior_art_only")

    # 3. Shuffled prior-art baseline
    shuffled_train = shuffle_prior_art(train, seed=42)
    shuffled_valid = shuffle_prior_art(valid, seed=43)
    save_baseline(shuffled_train, shuffled_valid, "shuffled_prior_art")


if __name__ == "__main__":
    main()