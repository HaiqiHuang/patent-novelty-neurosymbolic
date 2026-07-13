from rapidfuzz import fuzz

from enrich_patentmatch.utils.text import normalize_text


def locate_sentence_in_claims(source_sentence: str, claims: list[dict]) -> dict:
    """
    Locate which claim contains or best matches the source sentence.

    Returns:
        {
            "claim_no": int,
            "claim_text": str,
            "score": float
        }
    """
    source_norm = normalize_text(source_sentence)

    best_claim = None
    best_score = -1

    for claim in claims:
        claim_norm = normalize_text(claim["text"])

        score = fuzz.partial_ratio(source_norm, claim_norm)

        if score > best_score:
            best_score = score
            best_claim = claim

    return {
        "claim_no": best_claim["claim_no"],
        "claim_text": best_claim["text"],
        "score": best_score,
    }