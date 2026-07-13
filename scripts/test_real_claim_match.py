from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from enrich_patentmatch.epo.fetch_claims import fetch_claims_xml
from enrich_patentmatch.epo.parse_ops_xml import parse_claims_from_xml
from enrich_patentmatch.claims.match_claim import locate_sentence_in_claims
from enrich_patentmatch.claims.expand_paths import build_dependency_graph, expand_paths


def main():
    publication_number = "EP2873927A1"

    source_sentence = (
        "The indoor unit according to any one of the preceding claims wherein "
        "the material of the window insert 260 263 264 265 266 is translucent "
        "for light of a frequency irradiated by the LED 440 441 442 443 444 "
        "and translucent for the manipulation signal received by the receiving "
        "part 432 435."
    )

    xml = fetch_claims_xml(publication_number, use_cache=True)
    claims = parse_claims_from_xml(xml)

    match = locate_sentence_in_claims(source_sentence, claims)

    print("Matched claim:")
    print(f"Claim no: {match['claim_no']}")
    print(f"Score: {match['score']}")
    print()
    print(match["claim_text"])

    graph = build_dependency_graph(claims)
    paths = expand_paths(graph, match["claim_no"])

    print()
    print(f"Direct parents: {graph[match['claim_no']]['parents']}")
    print(f"Number of expanded paths: {len(paths)}")
    print("First 5 paths:")
    for path in paths[:5]:
        print(path)


if __name__ == "__main__":
    main()