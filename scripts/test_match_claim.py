from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from enrich_patentmatch.epo.parse_ops_xml import parse_claims_from_xml
from enrich_patentmatch.claims.match_claim import locate_sentence_in_claims


def main():
    xml_path = PROJECT_ROOT / "tests" / "fixtures" / "sample_claims_EP3045974A1.xml"
    xml = xml_path.read_text(encoding="utf-8")
    claims = parse_claims_from_xml(xml)

    source_sentence = (
        "Liquid food product obtained by the process according to any one of claims 1 to 13."
    )

    match = locate_sentence_in_claims(source_sentence, claims)

    print(match)

    assert match["claim_no"] == 14
    assert match["score"] > 95

    print("Claim matching test passed.")


if __name__ == "__main__":
    main()