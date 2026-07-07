import re


def clean_text(text: str) -> str:
    """Collapse whitespace and strip leading/trailing spaces."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_text(text: str) -> str:
    """
    Normalize patent text for fuzzy matching.

    This is intentionally aggressive:
    - lowercase
    - remove punctuation
    - normalize whitespace
    - treat hyphen as a separator
    """
    text = text.lower()
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()