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
    render_claim_path,
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

    graph = build_dependency_graph(claims)
    matched_claim_no = match["claim_no"]
    paths = expand_paths(graph, matched_claim_no)

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
        "expanded_paths": paths,
        "expanded_texts": [
            render_claim_path(graph, path)
            for path in paths
        ],
    }

    output_path = PROJECT_ROOT / "data" / "interim" / "real_sample_enriched.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(enriched, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote enriched sample to: {output_path}")
    print()
    print(f"Matched claim no: {matched_claim_no}")
    print(f"Match score: {match['score']}")
    print(f"Direct parents: {graph[matched_claim_no]['parents']}")
    print(f"Number of expanded paths: {len(paths)}")
    print()
    print("First expanded path:")
    if paths:
        print(paths[0])
        print(render_claim_path(graph, paths[0])[:1000])


if __name__ == "__main__":
    main()