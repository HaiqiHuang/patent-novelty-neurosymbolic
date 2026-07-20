import argparse
import time
from pathlib import Path
import sys

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from enrich_patentmatch.epo.fetch_claims import fetch_claims_xml


def normalize_publication_number(pub) -> str:
    if pd.isna(pub):
        return ""

    pub = str(pub).strip().upper()
    pub = pub.replace(" ", "")
    pub = pub.replace("-", "")
    return pub


def collect_unique_source_pubs(input_csvs: list[str]) -> list[str]:
    pubs = set()

    for path in input_csvs:
        df = pd.read_csv(path)
        for pub in df["source_pub"]:
            pub = normalize_publication_number(pub)
            if pub:
                pubs.add(pub)

    return sorted(pubs)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_csvs",
        type=str,
        nargs="+",
        required=True,
        help="One or more processed_v2 CSV files.",
    )
    parser.add_argument(
        "--sleep_seconds",
        type=float,
        default=0.0,
        help="Optional sleep after each request.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only prefetch first N unique source publications.",
    )

    args = parser.parse_args()

    pubs = collect_unique_source_pubs(args.input_csvs)

    if args.limit is not None:
        pubs = pubs[: args.limit]

    print("Input CSVs:", args.input_csvs)
    print("Unique source_pub:", len(pubs))

    success = 0
    errors = 0

    error_path = Path("data/interim/prefetch_epo_claims_errors.txt")
    error_path.parent.mkdir(parents=True, exist_ok=True)

    with error_path.open("w", encoding="utf-8") as err_f:
        for pub in tqdm(pubs):
            try:
                xml = fetch_claims_xml(pub, use_cache=True)
                success += 1

                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)

            except Exception as exc:
                errors += 1
                err_f.write(f"{pub}\t{type(exc).__name__}\t{exc}\n")

    print()
    print("Done.")
    print("Success:", success)
    print("Errors:", errors)
    print("Error log:", error_path)


if __name__ == "__main__":
    main()