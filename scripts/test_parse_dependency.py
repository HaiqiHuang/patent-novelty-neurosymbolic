from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from enrich_patentmatch.epo.parse_ops_xml import parse_claims_from_xml
from enrich_patentmatch.claims.parse_dependency import parse_and_repair_dependency


def main():
    xml_path = PROJECT_ROOT / "tests" / "fixtures" / "sample_claims_EP3045974A1.xml"
    xml = xml_path.read_text(encoding="utf-8")

    claims = parse_claims_from_xml(xml)

    expected_parents = {
        1: [],
        2: [1],
        3: [1, 2],
        4: [3],
        5: [1, 2, 3, 4],
        6: [1, 2, 3, 4, 5],
        7: [1, 2, 3, 4, 5, 6],
        8: [1, 2, 3, 4, 5, 6, 7],
        9: [1, 2, 3, 4, 5, 6, 7, 8],
        10: [1, 2, 3, 4, 5, 6, 7, 8, 9],
        11: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        12: [11],
        13: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        14: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        15: [14],
        16: [14, 15],
        17: [14],
        18: [16],
    }

    print(f"Parsed {len(claims)} claims.")
    print()

    for claim in claims:
        claim_no = claim["claim_no"]
        claim_text = claim["text"]

        dep = parse_and_repair_dependency(claim_text, claim_no)

        print(f"Claim {claim_no}")
        print(f"Type: {dep['type']}")
        print(f"Expression: {dep['expression']}")
        print(f"Parents: {dep['parents']}")
        print("-" * 80)

        assert dep["parents"] == expected_parents[claim_no], (
            f"Claim {claim_no} failed. "
            f"Expected {expected_parents[claim_no]}, got {dep['parents']}"
        )

    print()
    print("All dependency parser tests passed.")


if __name__ == "__main__":
    main()