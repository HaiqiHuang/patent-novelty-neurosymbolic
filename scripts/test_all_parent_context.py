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
    collect_parent_claims_from_paths,
    render_claims_by_numbers,
    build_all_parent_context,
)


def main():
    publication_number = "EP2873927A1"

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

    xml = fetch_claims_xml(publication_number, use_cache=True)
    claims = parse_claims_from_xml(xml)

    print(f"Parsed claims: {len(claims)}")
    assert len(claims) > 0, "No claims parsed from XML."

    match = locate_sentence_in_claims(source_sentence, claims)
    matched_claim_no = match["claim_no"]

    print(f"Matched claim no: {matched_claim_no}")
    print(f"Match score: {match['score']}")
    print()

    assert matched_claim_no is not None, "No matched claim found."
    assert match["score"] >= 80, (
        f"Claim match score too low: {match['score']}. "
        "Please inspect match_claim.py or source_sentence."
    )

    graph = build_dependency_graph(claims)

    paths = expand_paths(
        graph=graph,
        claim_no=matched_claim_no,
        max_paths=2000,
        max_depth=20,
    )

    print(f"Expanded paths: {len(paths)}")
    assert len(paths) > 0, "No expanded paths generated."

    parent_claims = collect_parent_claims_from_paths(
        paths=paths,
        target_claim_no=matched_claim_no,
    )

    print(f"All parent claims: {parent_claims}")
    print(f"Number of parent claims: {len(parent_claims)}")
    print()

    assert matched_claim_no not in parent_claims, (
        "Target claim should not appear in parent claims."
    )

    # In this EP2873927A1 example, the matched claim should be a dependent claim.
    assert len(parent_claims) > 0, "Expected at least one parent claim."

    parent_context = render_claims_by_numbers(
        graph=graph,
        claim_numbers=parent_claims,
    )

    assert "Claim" in parent_context, "Rendered parent context seems empty."

    all_parent_result = build_all_parent_context(
        graph=graph,
        paths=paths,
        target_claim_no=matched_claim_no,
    )

    assert all_parent_result["target_claim_no"] == matched_claim_no
    assert all_parent_result["parent_claims"] == parent_claims
    assert "[CURRENT CLAIM]" in all_parent_result["all_parent_context"]
    assert "[ALL PARENT CLAIMS]" in all_parent_result["all_parent_context"]

    model_input_text = (
        all_parent_result["all_parent_context"]
        + "\n\n[CITED SENTENCE]\n"
        + cited_sentence
    )

    print("All-parent context preview:")
    print(model_input_text[:10000])
    print()

    print("All-parent context test passed.")


if __name__ == "__main__":
    main()