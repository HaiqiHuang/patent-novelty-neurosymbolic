import re
from bs4 import BeautifulSoup

from enrich_patentmatch.utils.text import clean_text


def _match_leading_claim_number(text: str) -> tuple[int | None, str]:
    """
    Match a leading claim number.

    Examples:
        '1. An indoor unit ...' -> (1, 'An indoor unit ...')
        '14. Liquid food product ...' -> (14, 'Liquid food product ...')
    """
    text = clean_text(text)

    m = re.match(r"^(\d+)\s*[\.\)]\s*(.*)$", text)
    if not m:
        return None, text

    claim_no = int(m.group(1))
    rest = m.group(2).strip()

    return claim_no, rest


def parse_claims_from_xml(xml: str) -> list[dict]:
    """
    Parse claims from EPO OPS XML or simplified local XML.

    Handles two common formats:

    1. Normal structured format:
        <claim>
          <claim-num>1.</claim-num>
          <claim-text>...</claim-text>
        </claim>

    2. EPO OPS text-only format:
        <claim>
          <claim-text>1. First claim ...</claim-text>
          <claim-text>continuation of first claim ...</claim-text>
          <claim-text>2. Second claim ...</claim-text>
          <claim-text>continuation of second claim ...</claim-text>
        </claim>
    """
    soup = BeautifulSoup(xml, "xml")

    parsed_claims: list[dict] = []

    for claim in soup.find_all("claim"):
        claim_text_tags = claim.find_all("claim-text")
        if not claim_text_tags:
            continue

        claim_num_tag = claim.find("claim-num")
        claim_id = claim.get("id", "")

        # Case A: normal structured claim with <claim-num>
        if claim_num_tag:
            num_match = re.search(r"\d+", claim_num_tag.get_text(" "))
            if not num_match:
                continue

            claim_no = int(num_match.group())
            full_text = clean_text(
                " ".join(tag.get_text(" ") for tag in claim_text_tags)
            )

            parsed_claims.append(
                {
                    "claim_no": claim_no,
                    "text": full_text,
                }
            )
            continue

        # Case B: claim id like CLM-00001 and only one actual claim inside
        # We still prefer text-based splitting below, because EPO OPS may put
        # multiple numbered claims inside a single <claim>.
        text_parts = [
            clean_text(tag.get_text(" "))
            for tag in claim_text_tags
            if clean_text(tag.get_text(" "))
        ]

        current_claim_no = None
        current_parts: list[str] = []

        for part in text_parts:
            detected_no, rest = _match_leading_claim_number(part)

            if detected_no is not None:
                # Save previous claim before starting a new one.
                if current_claim_no is not None and current_parts:
                    parsed_claims.append(
                        {
                            "claim_no": current_claim_no,
                            "text": clean_text(" ".join(current_parts)),
                        }
                    )

                current_claim_no = detected_no
                current_parts = [rest]
            else:
                # Continuation of current claim.
                if current_claim_no is not None:
                    current_parts.append(part)
                else:
                    # Fallback: no detected number yet.
                    # Try claim id if available.
                    num_match = re.search(r"\d+", claim_id)
                    if num_match:
                        current_claim_no = int(num_match.group())
                        current_parts = [part]

        # Save final claim.
        if current_claim_no is not None and current_parts:
            parsed_claims.append(
                {
                    "claim_no": current_claim_no,
                    "text": clean_text(" ".join(current_parts)),
                }
            )

    # Deduplicate by claim number, keeping the longest text if duplicates occur.
    by_no: dict[int, str] = {}
    for claim in parsed_claims:
        claim_no = claim["claim_no"]
        text = claim["text"]

        if claim_no not in by_no or len(text) > len(by_no[claim_no]):
            by_no[claim_no] = text

    claims = [
        {
            "claim_no": claim_no,
            "text": text,
        }
        for claim_no, text in by_no.items()
    ]

    claims.sort(key=lambda x: x["claim_no"])
    return claims