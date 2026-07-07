import re
from bs4 import BeautifulSoup

from enrich_patentmatch.utils.text import clean_text


def parse_claims_from_xml(xml: str) -> list[dict]:
    """
    Parse claims from an EPO-like XML string.

    Returns:
        [
            {"claim_no": 1, "text": "..."},
            {"claim_no": 2, "text": "..."},
        ]
    """
    soup = BeautifulSoup(xml, "xml")
    claims = []

    for claim in soup.find_all("claim"):
        claim_id = claim.get("id", "")

        claim_num_tag = claim.find("claim-num")

        if claim_num_tag:
            num_match = re.search(r"\d+", claim_num_tag.get_text(" "))
        else:
            num_match = re.search(r"\d+", claim_id)

        if not num_match:
            continue

        claim_no = int(num_match.group())
        claim_text = clean_text(claim.get_text(" "))

        claims.append(
            {
                "claim_no": claim_no,
                "text": claim_text,
            }
        )

    claims.sort(key=lambda x: x["claim_no"])
    return claims