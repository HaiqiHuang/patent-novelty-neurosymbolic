import argparse
import json
from pathlib import Path

import pandas as pd


def convert_jsonl_to_csv(
    input_jsonl: str,
    output_csv: str,
) -> None:
    input_path = Path(input_jsonl)
    output_path = Path(output_csv)

    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            rows.append(
                {
                    # 训练脚本主要用这三列
                    # 这里的 patent_application 已经不是原始 claim sentence，
                    # 而是 current claim + all parent claims + cited sentence
                    "patent_application": item["all_parent_context"],
                    "prior_art": item["prior_art"],
                    "novelty": item["novelty"],

                    # 以下是 debug / analysis columns
                    "source_pub": item.get("source_pub"),
                    "cited_pub": item.get("cited_pub"),
                    "claim_id": item.get("claim_id"),
                    "date": item.get("date"),

                    "matched_claim_no": item.get("matched_claim_no"),
                    "match_score": item.get("match_score"),
                    "match_warning": item.get("match_warning"),

                    "direct_parent_claims": json.dumps(
                        item.get("direct_parent_claims", []),
                        ensure_ascii=False,
                    ),
                    "all_parent_claims": json.dumps(
                        item.get("all_parent_claims", []),
                        ensure_ascii=False,
                    ),
                    "num_all_parent_claims": item.get(
                        "num_all_parent_claims", 0
                    ),

                    "missing_parent_claims": json.dumps(
                        item.get("missing_parent_claims", []),
                        ensure_ascii=False,
                    ),
                    "num_missing_parent_claims": item.get(
                        "num_missing_parent_claims", 0
                    ),

                    "num_expanded_paths": item.get("num_expanded_paths"),
                    "num_claims_in_publication": item.get(
                        "num_claims_in_publication"
                    ),
                    "input_mode": item.get(
                        "input_mode",
                        "current_claim_plus_all_parents",
                    ),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Rows: {len(df)}")
    print()
    print("Columns:")
    print(df.columns.tolist())
    print()
    print("Label counts:")
    print(df["novelty"].value_counts())
    print()
    print("Preview:")
    print(df.head(2))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_jsonl",
        type=str,
        required=True,
        help="Input all-parent enriched JSONL file.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Output CSV file for model training.",
    )

    args = parser.parse_args()

    convert_jsonl_to_csv(
        input_jsonl=args.input_jsonl,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()