import re
from typing import Any


def parse_number_expression(expr: str) -> list[int]:
    """
    Parse claim number expressions.

    Examples:
        "1" -> [1]
        "1 or 2" -> [1, 2]
        "1, 2, or 3" -> [1, 2, 3]
        "1 to 4" -> [1, 2, 3, 4]
        "1-4" -> [1, 2, 3, 4]
        "1 through 4" -> [1, 2, 3, 4]
    """
    expr = expr.lower()
    expr = expr.replace("–", "-").replace("—", "-")
    expr = expr.replace("through", "to")

    nums: set[int] = set()

    # Ranges: 1-4, 1 to 4
    range_spans = []
    for m in re.finditer(r"(\d+)\s*(?:-|to)\s*(\d+)", expr):
        start, end = int(m.group(1)), int(m.group(2))
        range_spans.append(m.span())

        if start <= end:
            nums.update(range(start, end + 1))

    # Standalone numbers.
    # It is okay if this re-adds range endpoints because nums is a set.
    for m in re.finditer(r"\d+", expr):
        nums.add(int(m.group()))

    return sorted(nums)


def _extract_claim_reference_matches(text: str) -> list[tuple[str, list[int]]]:
    """
    Extract explicit claim references from normalized claim text.

    Handles:
        claim 1
        claims 1 or 2
        claims 1 to 4
        any one of claims 1 to 13
        defined in claim 15
        of claim 14

    Returns:
        [
            ("claims 1 to 4", [1,2,3,4]),
            ("claim 14", [14])
        ]
    """
    matches: list[tuple[str, list[int]]] = []

    # Capture "claim 1", "claims 1 or 2", "claims 1 to 4", etc.
    # Stop before common claim-body words like wherein/comprising.
    pattern = re.compile(
        r"\bclaims?\s+"
        r"("
        r"\d+"
        r"(?:\s*(?:,|or|and|-|to|through)\s*\d+)*"
        r")",
        flags=re.IGNORECASE,
    )

    for m in pattern.finditer(text):
        full_expr = m.group(0)
        number_expr = m.group(1)
        parents = parse_number_expression(number_expr)
        matches.append((full_expr, parents))

    return matches


def repair_squashed_range_dependency(dep: dict[str, Any], current_claim_no: int) -> dict[str, Any]:
    """
    Repair cases caused by lost hyphen:
        Claims 13 -> Claims 1-3

    Only repairs when the parsed parent is impossible or suspicious,
    e.g. Claim 4 depending on Claim 13.
    """
    parents = dep.get("parents", [])

    if len(parents) != 1:
        return dep

    ref = parents[0]

    # If referenced claim is before current claim, it is plausible.
    if ref < current_claim_no:
        return dep

    ref_str = str(ref)

    # Example: 13 -> 1-3
    if len(ref_str) == 2:
        start = int(ref_str[0])
        end = int(ref_str[1])

        if start < end and end < current_claim_no:
            repaired = dep.copy()
            repaired["parents"] = list(range(start, end + 1))
            repaired["expression_repaired"] = f"claims {start}-{end}"
            repaired["warning"] = (
                f"Suspicious dependency claim {ref}; "
                f"repaired as claims {start}-{end}"
            )
            return repaired

    return dep


def parse_dependency(claim_text: str, current_claim_no: int) -> dict[str, Any]:
    """
    Parse direct claim dependencies from one claim.

    Returns:
        {
            "expression": "...",
            "parents": [...],
            "type": "independent" | "explicit" | "any_preceding" | "multiple_explicit"
        }
    """
    text = claim_text.lower()
    text = text.replace("–", "-").replace("—", "-")

    # "any one of the preceding claims"
    if re.search(r"\b(any\s+one\s+of\s+)?(the\s+)?preceding\s+claims?\b", text):
        return {
            "expression": "preceding claims",
            "parents": list(range(1, current_claim_no)),
            "type": "any_preceding",
        }

    # "any preceding claim"
    if re.search(r"\bany\s+preceding\s+claim\b", text):
        return {
            "expression": "any preceding claim",
            "parents": list(range(1, current_claim_no)),
            "type": "any_preceding",
        }

    explicit_matches = _extract_claim_reference_matches(text)

    if explicit_matches:
        expressions = []
        parents_set: set[int] = set()

        for expr, parents in explicit_matches:
            expressions.append(expr)
            parents_set.update(parents)

        parents = sorted(parents_set)

        dep_type = "explicit" if len(explicit_matches) == 1 else "multiple_explicit"

        return {
            "expression": "; ".join(expressions),
            "parents": parents,
            "type": dep_type,
        }

    return {
        "expression": None,
        "parents": [],
        "type": "independent",
    }


def parse_and_repair_dependency(claim_text: str, current_claim_no: int) -> dict[str, Any]:
    """
    Parse dependency, repair obvious OCR/text-cleaning errors, and remove invalid parents.
    """
    dep = parse_dependency(claim_text, current_claim_no)
    dep = repair_squashed_range_dependency(dep, current_claim_no)

    dep = dep.copy()
    dep["parents"] = [
        p for p in dep.get("parents", [])
        if 0 < p < current_claim_no
    ]

    return dep