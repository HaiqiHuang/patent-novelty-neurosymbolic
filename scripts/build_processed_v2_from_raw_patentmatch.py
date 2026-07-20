import argparse
from pathlib import Path

import pandas as pd


RAW_COLUMN_MAP = {
    "source_pub": "patent_application_id",
    "cited_pub": "cited_document_id",
    "patent_application": "text",
    "prior_art": "text_b",
    "novelty": "label",
}


PROCESSED_COLUMN_MAP = {
    "patent_application": "patent_application",
    "prior_art": "prior_art",
    "novelty": "novelty",
}


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""

    text = str(value)
    text = " ".join(text.split())
    return text


def normalize_publication_number(value) -> str:
    if value is None or pd.isna(value):
        return ""

    pub = str(value).strip().upper()
    pub = pub.replace(" ", "")
    pub = pub.replace("-", "")
    return pub


def make_key_df(
    df: pd.DataFrame,
    text_a_col: str,
    text_b_col: str,
    label_col: str,
) -> pd.DataFrame:
    key_df = df.copy()

    key_df["_key_patent_application"] = key_df[text_a_col].apply(normalize_text)
    key_df["_key_prior_art"] = key_df[text_b_col].apply(normalize_text)
    key_df["_key_novelty"] = key_df[label_col].astype(int)

    return key_df


def load_raw_tsv(path: str) -> pd.DataFrame:
    raw_path = Path(path)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw TSV not found: {raw_path}")

    df = pd.read_csv(raw_path, sep="\t")

    required_cols = [
        RAW_COLUMN_MAP["source_pub"],
        RAW_COLUMN_MAP["cited_pub"],
        RAW_COLUMN_MAP["patent_application"],
        RAW_COLUMN_MAP["prior_art"],
        RAW_COLUMN_MAP["novelty"],
    ]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing raw columns in {raw_path}: {missing}\n"
            f"Available raw columns: {df.columns.tolist()}"
        )

    return df


def load_processed_csv(path: str) -> pd.DataFrame:
    processed_path = Path(path)

    if not processed_path.exists():
        raise FileNotFoundError(f"Processed CSV not found: {processed_path}")

    df = pd.read_csv(processed_path)

    required_cols = [
        PROCESSED_COLUMN_MAP["patent_application"],
        PROCESSED_COLUMN_MAP["prior_art"],
        PROCESSED_COLUMN_MAP["novelty"],
    ]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing processed columns in {processed_path}: {missing}\n"
            f"Available processed columns: {df.columns.tolist()}"
        )

    return df


def prepare_raw_for_join(raw_df: pd.DataFrame) -> pd.DataFrame:
    raw_keyed = make_key_df(
        raw_df,
        text_a_col=RAW_COLUMN_MAP["patent_application"],
        text_b_col=RAW_COLUMN_MAP["prior_art"],
        label_col=RAW_COLUMN_MAP["novelty"],
    )

    out = pd.DataFrame()

    out["_key_patent_application"] = raw_keyed["_key_patent_application"]
    out["_key_prior_art"] = raw_keyed["_key_prior_art"]
    out["_key_novelty"] = raw_keyed["_key_novelty"]

    out["source_pub"] = raw_keyed[RAW_COLUMN_MAP["source_pub"]].apply(
        normalize_publication_number
    )
    out["cited_pub"] = raw_keyed[RAW_COLUMN_MAP["cited_pub"]].apply(
        normalize_publication_number
    )

    out["patent_application"] = raw_keyed[RAW_COLUMN_MAP["patent_application"]].astype(str)
    out["prior_art"] = raw_keyed[RAW_COLUMN_MAP["prior_art"]].astype(str)
    out["novelty"] = raw_keyed[RAW_COLUMN_MAP["novelty"]].astype(int)

    # keep claim_id / date for later debug
    if "claim_id" in raw_keyed.columns:
        out["claim_id"] = raw_keyed["claim_id"].astype(str)

    if "date" in raw_keyed.columns:
        out["date"] = raw_keyed["date"]

    # 如果同一个 text pair 重复出现，先保留第一条。
    # 注意：这里不 drop 掉 processed 内部样本，只是避免 merge 时一条样本膨胀成多条。
    dedup_subset = [
        "_key_patent_application",
        "_key_prior_art",
        "_key_novelty",
    ]

    before = len(out)
    out = out.drop_duplicates(subset=dedup_subset, keep="first").copy()
    after = len(out)

    print(f"Raw rows before dedup: {before}")
    print(f"Raw rows after dedup: {after}")

    return out


def prepare_processed_split_for_join(
    processed_df: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    keyed = make_key_df(
        processed_df,
        text_a_col=PROCESSED_COLUMN_MAP["patent_application"],
        text_b_col=PROCESSED_COLUMN_MAP["prior_art"],
        label_col=PROCESSED_COLUMN_MAP["novelty"],
    )

    out = pd.DataFrame()

    out["_key_patent_application"] = keyed["_key_patent_application"]
    out["_key_prior_art"] = keyed["_key_prior_art"]
    out["_key_novelty"] = keyed["_key_novelty"]

    out["split"] = split_name
    out["processed_row_index"] = range(len(out))

    return out


def build_one_split(
    raw_prepared: pd.DataFrame,
    processed_split_path: str,
    split_name: str,
    output_path: str,
) -> None:
    processed_df = load_processed_csv(processed_split_path)

    split_keys = prepare_processed_split_for_join(
        processed_df=processed_df,
        split_name=split_name,
    )

    merged = split_keys.merge(
        raw_prepared,
        on=[
            "_key_patent_application",
            "_key_prior_art",
            "_key_novelty",
        ],
        how="left",
        validate="many_to_one",
    )

    missing = merged[merged["source_pub"].isna()].copy()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if len(missing) > 0:
        missing_path = output.with_suffix(".missing_pub.csv")
        missing.to_csv(missing_path, index=False)
        print(f"[WARN] {split_name}: missing publication numbers: {len(missing)}")
        print(f"[WARN] Missing rows written to: {missing_path}")

    ok = merged[~merged["source_pub"].isna()].copy()

    final_cols = [
        "source_pub",
        "cited_pub",
        "patent_application",
        "prior_art",
        "novelty",
    ]

    if "claim_id" in ok.columns:
        final_cols.append("claim_id")

    if "date" in ok.columns:
        final_cols.append("date")

    final_df = ok[final_cols + ["processed_row_index"]].copy()
    final_df = final_df.sort_values("processed_row_index")
    final_df = final_df.drop(columns=["processed_row_index"])

    final_df.to_csv(output, index=False)

    print("=" * 100)
    print(f"Split: {split_name}")
    print(f"Old processed input: {processed_split_path}")
    print(f"New processed_v2 output: {output}")
    print(f"Rows in old processed: {len(processed_df)}")
    print(f"Rows written to processed_v2: {len(final_df)}")
    print(f"Rows missing source_pub: {len(missing)}")
    print()
    print(final_df.head(2))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw_train_tsv",
        type=str,
        default="data/raw/patentmatch_train_ultrabalanced.tsv",
    )
    parser.add_argument(
        "--raw_test_tsv",
        type=str,
        default="data/raw/patentmatch_test_ultrabalanced.tsv",
    )

    parser.add_argument(
        "--processed_train_csv",
        type=str,
        default="data/processed/train.csv",
    )
    parser.add_argument(
        "--processed_valid_csv",
        type=str,
        default="data/processed/valid.csv",
    )
    parser.add_argument(
        "--processed_test_csv",
        type=str,
        default="data/processed/test.csv",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed_v2",
    )

    args = parser.parse_args()

    raw_train = load_raw_tsv(args.raw_train_tsv)
    raw_test = load_raw_tsv(args.raw_test_tsv)

    print("=" * 100)
    print("Preparing raw train...")
    raw_train_prepared = prepare_raw_for_join(raw_train)

    print("=" * 100)
    print("Preparing raw test...")
    raw_test_prepared = prepare_raw_for_join(raw_test)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # processed train / valid come from raw train
    build_one_split(
        raw_prepared=raw_train_prepared,
        processed_split_path=args.processed_train_csv,
        split_name="train",
        output_path=str(output_dir / "train.csv"),
    )

    build_one_split(
        raw_prepared=raw_train_prepared,
        processed_split_path=args.processed_valid_csv,
        split_name="valid",
        output_path=str(output_dir / "valid.csv"),
    )

    # processed test comes from raw test
    build_one_split(
        raw_prepared=raw_test_prepared,
        processed_split_path=args.processed_test_csv,
        split_name="test",
        output_path=str(output_dir / "test.csv"),
    )


if __name__ == "__main__":
    main()