from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from enrich_patentmatch.epo.parse_ops_xml import parse_claims_from_xml


def main():
    xml_path = PROJECT_ROOT / "tests" / "fixtures" / "sample_claims_EP3045974A1.xml"
    xml = xml_path.read_text(encoding="utf-8")

    claims = parse_claims_from_xml(xml)

    print(f"Parsed {len(claims)} claims.")
    for claim in claims:
        print()
        print(f"Claim {claim['claim_no']}:")
        print(claim["text"])


if __name__ == "__main__":
    main()