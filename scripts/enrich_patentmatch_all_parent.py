import argparse
import json
import time
from pathlib import Path
import sys

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from enrich_patentmatch.epo.fetch_claims import fetch_claims_xml
from enrich_patentmatch.epo.parse_ops_xml import parse_claims_from_xml
from enrich_patentmatch.claims.match_claim import locate_sentence_in_claims
from enrich_patentmatch.claims.expand_paths import (
    build_dependency_graph,
    expand_paths,
    build_all_parent_context,
)


COLUMN_MAP = {
    "source_pub": "source_pub",
    "cited_pub": "cited_pub",
    "source_sentence": "patent_application",
    "cited_sentence": "prior_art",
    "label": "novelty",
}


def normalize_publication_number(pub) -> str:
    if pd.isna(pub):
        return ""

    pub = str(pub).strip().upper()
    pub = pub.replace(" ", "")
    pub = pub.replace("-", "")
    return pub


def get_value(row, col_name: str, default=None):
    if col_name not in row:
        return default

    value = row[col_name]

    if pd.isna(value):
        return default

    return value


def load_publication_data(
    source_pub: str,
    publication_cache: dict,
    sleep_seconds: float = 0.0,
    verbose: bool = False,
) -> tuple[list[dict], dict]:
    """
    Load claims and dependency graph for one source publication.

    Important:
        This function caches by source_pub in memory, so the same patent is
        fetched / parsed / graphed only once during one script run.
    """
    if source_pub in publication_cache:
        if verbose:
            print(f"[cache hit] {source_pub}", flush=True)
        return publication_cache[source_pub]["claims"], publication_cache[source_pub]["graph"]

    if verbose:
        print(f"[cache miss] fetching {source_pub}", flush=True)

    xml = fetch_claims_xml(source_pub, use_cache=True)

    if sleep_seconds > 0:
        time.sleep(sleep_seconds)

    claims = parse_claims_from_xml(xml)

    if not claims:
        raise ValueError(f"No claims parsed for {source_pub}")

    graph = build_dependency_graph(claims)

    publication_cache[source_pub] = {
        "claims": claims,
        "graph": graph,
    }

    if verbose:
        print(
            f"[loaded] {source_pub}: claims={len(claims)}, cache_size={len(publication_cache)}",
            flush=True,
        )

    return claims, graph


def enrich_one_row(
    row,
    row_index: int,
    publication_cache: dict,
    max_paths: int = 2000,
    max_depth: int = 20,
    min_match_score: float = 80.0,
    sleep_seconds: float = 0.0,
    verbose: bool = False,
) -> tuple[dict | None, dict | None]:
    source_pub = normalize_publication_number(
        get_value(row, COLUMN_MAP["source_pub"], "")
    )
    cited_pub = normalize_publication_number(
        get_value(row, COLUMN_MAP["cited_pub"], "")
    )

    source_sentence = get_value(row, COLUMN_MAP["source_sentence"], "")
    cited_sentence = get_value(row, COLUMN_MAP["cited_sentence"], "")
    label = get_value(row, COLUMN_MAP["label"], None)

    claim_id = get_value(row, "claim_id", None)
    date = get_value(row, "date", None)

    if not source_pub:
        return None, {
            "row_index": int(row_index),
            "claim_id": None if claim_id is None else str(claim_id),
            "error_type": "missing_source_pub",
            "message": "Missing source publication number.",
        }

    if not source_sentence:
        return None, {
            "row_index": int(row_index),
            "source_pub": source_pub,
            "claim_id": None if claim_id is None else str(claim_id),
            "error_type": "missing_source_sentence",
            "message": "Missing source sentence.",
        }

    try:
        if verbose:
            print(f"[row {row_index}] source_pub={source_pub}", flush=True)

        claims, graph = load_publication_data(
            source_pub=source_pub,
            publication_cache=publication_cache,
            sleep_seconds=sleep_seconds,
            verbose=verbose,
        )

        match = locate_sentence_in_claims(source_sentence, claims)
        matched_claim_no = match["claim_no"]
        match_score = match["score"]

        if verbose:
            print(
                f"[row {row_index}] matched claim={matched_claim_no}, "
                f"score={match_score}",
                flush=True,
            )

        if matched_claim_no not in graph:
            return None, {
                "row_index": int(row_index),
                "source_pub": source_pub,
                "cited_pub": cited_pub,
                "claim_id": None if claim_id is None else str(claim_id),
                "matched_claim_no": matched_claim_no,
                "match_score": match_score,
                "error_type": "matched_claim_not_in_graph",
                "message": "Matched claim number is not present in dependency graph.",
            }

        paths = expand_paths(
            graph=graph,
            claim_no=matched_claim_no,
            max_paths=max_paths,
            max_depth=max_depth,
        )

        all_parent = build_all_parent_context(
            graph=graph,
            paths=paths,
            target_claim_no=matched_claim_no,
        )

        model_input = (
            all_parent["all_parent_context"]
            + "\n\n[CITED SENTENCE]\n"
            + str(cited_sentence)
        )

        result = {
            "row_index": int(row_index),
            "claim_id": None if claim_id is None else str(claim_id),
            "date": None if date is None else str(date),

            "source_pub": source_pub,
            "cited_pub": cited_pub,
            "patent_application": str(source_sentence),
            "prior_art": str(cited_sentence),
            "novelty": None if label is None else int(label),

            "matched_claim_no": int(matched_claim_no),
            "match_score": float(match_score),
            "match_warning": None
            if match_score >= min_match_score
            else "low_match_score",
            "matched_claim_text": match["claim_text"],

            "dependency": graph[matched_claim_no]["dependency"],
            "direct_parent_claims": graph[matched_claim_no]["parents"],

            "num_claims_in_publication": len(claims),
            "num_expanded_paths": len(paths),
            "max_paths": max_paths,
            "max_depth": max_depth,

            "all_parent_claims": all_parent["parent_claims"],
            "num_all_parent_claims": len(all_parent["parent_claims"]),
            "missing_parent_claims": all_parent.get("missing_parent_claims", []),
            "num_missing_parent_claims": len(all_parent.get("missing_parent_claims", [])),
            "current_claim_text": all_parent["current_claim_text"],
            "parent_context": all_parent["parent_context"],
            "all_parent_context": all_parent["all_parent_context"],

            "model_input_current_plus_all_parents": model_input,
            "input_mode": "current_claim_plus_all_parents",
        }

        return result, None

    except Exception as e:
        return None, {
            "row_index": int(row_index),
            "source_pub": source_pub,
            "cited_pub": cited_pub,
            "claim_id": None if claim_id is None else str(claim_id),
            "error_type": type(e).__name__,
            "message": str(e),
        }


def enrich_file(
    input_csv: str,
    output_jsonl: str,
    error_jsonl: str,
    limit: int | None = None,
    max_paths: int = 2000,
    max_depth: int = 20,
    min_match_score: float = 80.0,
    sleep_seconds: float = 0.0,
    verbose: bool = False,
):
    input_path = Path(input_csv)
    output_path = Path(output_jsonl)
    error_path = Path(error_jsonl)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    if limit is not None:
        df = df.head(limit)

    unique_pubs = df[COLUMN_MAP["source_pub"]].nunique()

    print(f"Input: {input_path}", flush=True)
    print(f"Rows: {len(df)}", flush=True)
    print(f"Unique source_pub: {unique_pubs}", flush=True)
    print(f"Output: {output_path}", flush=True)
    print(f"Errors: {error_path}", flush=True)
    print(flush=True)

    publication_cache = {}

    success_count = 0
    error_count = 0

    with output_path.open("w", encoding="utf-8") as out_f, error_path.open(
        "w", encoding="utf-8"
    ) as err_f:
        for row_index, row in tqdm(df.iterrows(), total=len(df)):
            result, error = enrich_one_row(
                row=row,
                row_index=row_index,
                publication_cache=publication_cache,
                max_paths=max_paths,
                max_depth=max_depth,
                min_match_score=min_match_score,
                sleep_seconds=sleep_seconds,
                verbose=verbose,
            )

            if result is not None:
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                success_count += 1
            else:
                err_f.write(json.dumps(error, ensure_ascii=False) + "\n")
                error_count += 1

    print()
    print("Done.", flush=True)
    print(f"Success: {success_count}", flush=True)
    print(f"Errors: {error_count}", flush=True)
    print(f"Publication cache size: {len(publication_cache)}", flush=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--error_jsonl", type=str, required=True)

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_paths", type=int, default=2000)
    parser.add_argument("--max_depth", type=int, default=20)
    parser.add_argument("--min_match_score", type=float, default=80.0)
    parser.add_argument("--sleep_seconds", type=float, default=0.0)

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print row-level debug information.",
    )

    args = parser.parse_args()

    enrich_file(
        input_csv=args.input_csv,
        output_jsonl=args.output_jsonl,
        error_jsonl=args.error_jsonl,
        limit=args.limit,
        max_paths=args.max_paths,
        max_depth=args.max_depth,
        min_match_score=args.min_match_score,
        sleep_seconds=args.sleep_seconds,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()