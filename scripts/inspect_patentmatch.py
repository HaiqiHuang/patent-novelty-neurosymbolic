from pathlib import Path

import pandas as pd


def main() -> None:
    raw_dir = Path("data/raw")
    tsv_files = list(raw_dir.glob("*.tsv"))

    print("Found TSV files:")
    for file in tsv_files:
        print(f"- {file}")

    for file in tsv_files:
        print("\n" + "=" * 80)
        print(f"File: {file}")

        df = pd.read_csv(file, sep="\t")

        print("Shape:", df.shape)
        print("\nColumns:")
        print(df.columns.tolist())

        print("\nLabel distribution:")
        print(df["label"].value_counts())

        print("\nMissing values:")
        print(df[["text", "text_b", "label"]].isna().sum())

        print("\nExample:")
        print("Claim:")
        print(df.loc[0, "text"][:500])
        print("\nPrior art paragraph:")
        print(df.loc[0, "text_b"][:500])
        print("\nLabel:")
        print(df.loc[0, "label"])


if __name__ == "__main__":
    main()