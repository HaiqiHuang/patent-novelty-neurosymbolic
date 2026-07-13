from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from enrich_patentmatch.epo.fetch_claims import fetch_claims_xml
from enrich_patentmatch.epo.parse_ops_xml import parse_claims_from_xml


def main():
    publication_number = "EP2873927A1"

    xml = fetch_claims_xml(publication_number, use_cache=False)

    print("Successfully fetched claims XML.")
    print(f"Publication: {publication_number}")
    print(f"XML length: {len(xml)} characters")
    print()
    print("First 1000 characters:")
    print(xml[:1000])

    output_path = PROJECT_ROOT / "data" / "external" / "epo_claims" / f"{publication_number}.xml"
    print()
    print(f"Saved to: {output_path}")

    claims = parse_claims_from_xml(xml)
    print()
    print(f"Parsed {len(claims)} claims.")
    for claim in claims[:3]:
        print()
        print(f"Claim {claim['claim_no']}:")
        print(claim["text"][:500])


if __name__ == "__main__":
    main()