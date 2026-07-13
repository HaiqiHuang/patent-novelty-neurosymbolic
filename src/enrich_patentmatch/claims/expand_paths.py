from enrich_patentmatch.claims.parse_dependency import parse_and_repair_dependency


def build_dependency_graph(claims: list[dict]) -> dict:
    """
    Build a dependency graph from parsed claims.

    Returns:
        {
            1: {"text": "...", "parents": [], "dependency": {...}},
            2: {"text": "...", "parents": [1], "dependency": {...}},
        }
    """
    graph = {}

    for claim in claims:
        claim_no = claim["claim_no"]
        claim_text = claim["text"]

        dep = parse_and_repair_dependency(claim_text, claim_no)

        graph[claim_no] = {
            "text": claim_text,
            "parents": dep["parents"],
            "dependency": dep,
        }

    return graph


def expand_paths(graph: dict, claim_no: int, visited: set[int] | None = None) -> list[list[int]]:
    """
    Expand all valid dependency paths ending at claim_no.

    Example:
        Claim 4 depends on [1,2,3]
        -> [[1,4], [1,2,4], [1,3,4]]
        depending on parents' own dependencies.
    """
    if visited is None:
        visited = set()

    if claim_no in visited:
        raise ValueError(f"Cycle detected at claim {claim_no}")

    visited = visited | {claim_no}

    parents = graph[claim_no]["parents"]

    if not parents:
        return [[claim_no]]

    paths = []

    for parent in parents:
        parent_paths = expand_paths(graph, parent, visited)
        for path in parent_paths:
            paths.append(path + [claim_no])

    return paths


def render_claim_path(graph: dict, path: list[int]) -> str:
    parts = []

    for claim_no in path:
        parts.append(f"Claim {claim_no}: {graph[claim_no]['text']}")

    return "\n".join(parts)