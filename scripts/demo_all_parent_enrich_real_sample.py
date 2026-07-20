import json
from pathlib import Path
import sys

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


def main():
    source_pub = "EP2873927A1"
    cited_pub = "EP2363653"

    source_sentence = (
        "The indoor unit according to any one of the preceding claims wherein "
        "the material of the window insert 260 263 264 265 266 is translucent "
        "for light of a frequency irradiated by the LED 440 441 442 443 444 "
        "and translucent for the manipulation signal received by the receiving "
        "part 432 435."
    )

    cited_sentence = (
        "While the ECO lamp 20 is ON when the user presses the ECO advice button "
        "210 of the remote controller 200 the content of the energy saving advice "
        "is displayed onthe guidance display 220 of the remote controller 200."
    )

    xml = fetch_claims_xml(source_pub, use_cache=True)
    claims = parse_claims_from_xml(xml)

    match = locate_sentence_in_claims(source_sentence, claims)
    matched_claim_no = match["claim_no"]

    graph = build_dependency_graph(claims)

    paths = expand_paths(
        graph=graph,
        claim_no=matched_claim_no,
        max_paths=2000,
        max_depth=20,
    )

    all_parent = build_all_parent_context(
        graph=graph,
        paths=paths,
        target_claim_no=matched_claim_no,
    )

    enriched = {
        "source_pub": source_pub,
        "cited_pub": cited_pub,
        "source_sentence": source_sentence,
        "cited_sentence": cited_sentence,
        "matched_claim_no": matched_claim_no,
        "match_score": match["score"],
        "matched_claim_text": match["claim_text"],
        "dependency": graph[matched_claim_no]["dependency"],
        "direct_parent_claims": graph[matched_claim_no]["parents"],
        "num_expanded_paths": len(paths),
        "all_parent_claims": all_parent["parent_claims"],
        "current_claim_text": all_parent["current_claim_text"],
        "parent_context": all_parent["parent_context"],
        "all_parent_context": all_parent["all_parent_context"],
        "model_input_current_plus_all_parents": (
            all_parent["all_parent_context"]
            + "\n\n[CITED SENTENCE]\n"
            + cited_sentence
        ),
    }

    output_path = (
        PROJECT_ROOT
        / "data"
        / "interim"
        / "real_sample_all_parent_enriched.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(enriched, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote: {output_path}")
    print()
    print(f"Matched claim no: {matched_claim_no}")
    print(f"Match score: {match['score']}")
    print(f"Direct parents: {graph[matched_claim_no]['parents']}")
    print(f"All parent claims: {all_parent['parent_claims']}")
    print(f"Number of expanded paths: {len(paths)}")
    print()
    print("Model input preview:")
    print(enriched["model_input_current_plus_all_parents"][:10000])


if __name__ == "__main__":
    main()