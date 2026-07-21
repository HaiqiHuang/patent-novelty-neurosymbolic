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
        raise FileNotFoundError(
            f"Input JSONL not found: {input_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            rows.append(
                {
                    # Main training columns
                    "patent_application": item[
                        "patent_application"
                    ],
                    "prior_art": item["prior_art"],
                    "novelty": item["novelty"],

                    # Grouping fields required later
                    "split": item["split"],
                    "sample_id": item["sample_id"],
                    "path_id": item["path_id"],
                    "path_claim_numbers": json.dumps(
                        item.get(
                            "path_claim_numbers",
                            [],
                        ),
                        ensure_ascii=False,
                    ),
                    "path_depth": item.get(
                        "path_depth"
                    ),
                    "num_paths_for_sample": item.get(
                        "num_paths_for_sample"
                    ),

                    # Debug / analysis columns
                    "row_index": item.get("row_index"),
                    "source_pub": item.get("source_pub"),
                    "cited_pub": item.get("cited_pub"),
                    "claim_id": item.get("claim_id"),
                    "date": item.get("date"),

                    "matched_claim_no": item.get(
                        "matched_claim_no"
                    ),
                    "match_score": item.get(
                        "match_score"
                    ),
                    "match_warning": item.get(
                        "match_warning"
                    ),

                    "direct_parent_claims": json.dumps(
                        item.get(
                            "direct_parent_claims",
                            [],
                        ),
                        ensure_ascii=False,
                    ),
                    "missing_parent_claims": json.dumps(
                        item.get(
                            "missing_parent_claims",
                            [],
                        ),
                        ensure_ascii=False,
                    ),

                    "num_claims_in_publication": item.get(
                        "num_claims_in_publication"
                    ),
                    "num_expanded_paths": item.get(
                        "num_expanded_paths"
                    ),
                    "input_mode": item.get(
                        "input_mode",
                        "pathwise_claim_context",
                    ),
                }
            )

    df = pd.DataFrame(rows)

    duplicate_count = df.duplicated(
        subset=["sample_id", "path_id"]
    ).sum()

    if duplicate_count:
        raise ValueError(
            "Duplicate sample_id/path_id pairs: "
            f"{duplicate_count}"
        )

    label_consistency = (
        df.groupby("sample_id")["novelty"]
        .nunique()
    )

    if label_consistency.max() > 1:
        raise ValueError(
            "One sample_id has multiple labels."
        )

    df.to_csv(output_path, index=False)

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Path-wise rows: {len(df)}")
    print(
        "Original samples:",
        df["sample_id"].nunique(),
    )
    print()
    print("Paths per sample:")
    print(
        df.groupby("sample_id").size().describe()
    )
    print()
    print("Label counts by original sample:")
    print(
        df.drop_duplicates("sample_id")[
            "novelty"
        ].value_counts()
    )
    print()
    print("Preview:")
    print(
        df[
            [
                "sample_id",
                "path_id",
                "path_claim_numbers",
                "path_depth",
                "novelty",
            ]
        ].head(10)
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_jsonl",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    convert_jsonl_to_csv(
        input_jsonl=args.input_jsonl,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()