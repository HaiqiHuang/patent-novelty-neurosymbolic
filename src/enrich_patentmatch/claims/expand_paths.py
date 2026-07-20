from enrich_patentmatch.claims.parse_dependency import parse_and_repair_dependency


def build_dependency_graph(claims: list[dict]) -> dict:
    """
    Build a dependency graph from parsed claims.

    Returns:
        {
            1: {
                "text": "...",
                "parents": [],
                "missing_parents": [],
                "dependency": {...},
            },
            2: {
                "text": "...",
                "parents": [1],
                "missing_parents": [],
                "dependency": {...},
            },
        }

    Notes:
        Some dependency parser outputs may refer to claim numbers that are not
        present in the parsed EPO claims.

        Example:
            Claim 10 says "according to any one of claims 1 to 9",
            but the parsed XML only contains claims 1, 3, 4, ..., 10.

        In that case, claim 2 is a missing parent. We should not crash with
        KeyError. Instead, we keep only valid parents and record missing ones.
    """
    graph = {}

    for claim in claims:
        claim_no = claim["claim_no"]
        claim_text = claim["text"]

        dep = parse_and_repair_dependency(claim_text, claim_no)

        graph[claim_no] = {
            "text": claim_text,
            "parents": dep["parents"],
            "missing_parents": [],
            "dependency": dep,
        }

    valid_claim_numbers = set(graph.keys())

    for claim_no, node in graph.items():
        original_parents = node["parents"]

        valid_parents = []
        missing_parents = []

        for parent in original_parents:
            if parent in valid_claim_numbers:
                valid_parents.append(parent)
            else:
                missing_parents.append(parent)

        node["parents"] = valid_parents
        node["missing_parents"] = missing_parents

    return graph


def expand_paths(
    graph: dict,
    claim_no: int,
    visited: set[int] | None = None,
    max_paths: int = 2000,
    max_depth: int = 20,
) -> list[list[int]]:
    """
    Expand valid dependency paths ending at claim_no, with safety limits.

    Example:
        Claim 4 depends on [1, 2, 3]
        -> [[1, 4], [1, 2, 4], [1, 3, 4]]
        depending on parents' own dependencies.

    Args:
        graph: Dependency graph built by build_dependency_graph().
        claim_no: Target claim number.
        visited: Claims already visited during recursion.
        max_paths: Maximum number of paths to return.
        max_depth: Maximum recursion depth / path depth.

    Why max_paths matters:
        Some claims can generate hundreds or thousands of paths.
        This cap prevents path explosion.

    Robustness:
        If claim_no is not in graph, return an empty list instead of raising
        KeyError. This prevents one noisy dependency from crashing the whole
        enrichment pipeline.
    """
    if claim_no not in graph:
        return []

    if visited is None:
        visited = set()

    if claim_no in visited:
        raise ValueError(f"Cycle detected at claim {claim_no}")

    # Safety limit: stop if the dependency chain becomes too deep.
    if len(visited) >= max_depth:
        return [[claim_no]]

    visited = visited | {claim_no}

    parents = graph[claim_no]["parents"]

    if not parents:
        return [[claim_no]]

    paths = []

    for parent in parents:
        # Defensive check. build_dependency_graph() should already remove
        # missing parents, but this keeps expand_paths safe if called with
        # another graph.
        if parent not in graph:
            continue

        parent_paths = expand_paths(
            graph=graph,
            claim_no=parent,
            visited=visited,
            max_paths=max_paths,
            max_depth=max_depth,
        )

        for path in parent_paths:
            paths.append(path + [claim_no])

            # Safety limit: stop once enough paths have been collected.
            if len(paths) >= max_paths:
                return paths

    # If all parents were missing or skipped, keep the target claim as a
    # standalone path rather than returning no path.
    if not paths:
        return [[claim_no]]

    return paths


def render_claim_path(graph: dict, path: list[int]) -> str:
    """
    Convert a claim path like [1, 3, 14] into readable text.

    Example:
        [1, 3, 14]
        ->
        Claim 1: ...
        Claim 3: ...
        Claim 14: ...

    Notes:
        Missing claim numbers are skipped instead of raising KeyError.
    """
    parts = []

    for claim_no in path:
        if claim_no not in graph:
            continue

        parts.append(f"Claim {claim_no}: {graph[claim_no]['text']}")

    return "\n".join(parts)


def collect_parent_claims_from_paths(
    paths: list[list[int]],
    target_claim_no: int,
) -> list[int]:
    """
    Collect all unique parent / ancestor claims appearing in expanded paths.

    Args:
        paths: Expanded dependency paths.
            Example: [[1, 14], [1, 2, 14], [1, 3, 14]]
        target_claim_no: The current/source claim number.
            Example: 14

    Returns:
        Sorted unique parent / ancestor claim numbers,
        excluding the target claim itself.

    Example:
        paths = [[1, 14], [1, 2, 14], [1, 3, 14]]
        target_claim_no = 14
        -> [1, 2, 3]
    """
    parent_claims = set()

    for path in paths:
        for claim_no in path:
            if claim_no != target_claim_no:
                parent_claims.add(claim_no)

    return sorted(parent_claims)


def render_claims_by_numbers(
    graph: dict,
    claim_numbers: list[int],
) -> str:
    """
    Render a list of claim numbers into readable text.

    Example:
        [1, 2, 3]
        ->
        Claim 1: ...
        Claim 2: ...
        Claim 3: ...

    Notes:
        Missing claim numbers are skipped instead of raising KeyError.
    """
    parts = []

    for claim_no in claim_numbers:
        if claim_no not in graph:
            continue

        parts.append(f"Claim {claim_no}: {graph[claim_no]['text']}")

    return "\n".join(parts)


def collect_missing_parent_claims(
    graph: dict,
    target_claim_no: int,
) -> list[int]:
    """
    Collect missing parent claim numbers for the target claim.

    This only records direct missing parents from the target claim's dependency
    parse. It is mainly used for debugging enrichment quality.
    """
    if target_claim_no not in graph:
        return []

    return sorted(set(graph[target_claim_no].get("missing_parents", [])))


def build_all_parent_context(
    graph: dict,
    paths: list[list[int]],
    target_claim_no: int,
) -> dict:
    """
    Build an all-parent context for a target claim.

    This is a simple baseline:
        current claim
        + all unique parent / ancestor claims appearing in expanded paths

    It does not select top-k paths and does not use MIL.

    Returns:
        {
            "target_claim_no": 14,
            "parent_claims": [1, 2, 3, ...],
            "missing_parent_claims": [7, ...],
            "current_claim_text": "...",
            "parent_context": "Claim 1: ...\nClaim 2: ...",
            "all_parent_context": "[CURRENT CLAIM]...\n[ALL PARENT CLAIMS]..."
        }
    """
    if target_claim_no not in graph:
        raise KeyError(target_claim_no)

    parent_claims = collect_parent_claims_from_paths(
        paths=paths,
        target_claim_no=target_claim_no,
    )

    # Keep only claim numbers that exist in the graph.
    parent_claims = [
        claim_no
        for claim_no in parent_claims
        if claim_no in graph
    ]

    missing_parent_claims = collect_missing_parent_claims(
        graph=graph,
        target_claim_no=target_claim_no,
    )

    current_claim_text = graph[target_claim_no]["text"]

    parent_context = render_claims_by_numbers(
        graph=graph,
        claim_numbers=parent_claims,
    )

    all_parent_context = (
        "[CURRENT CLAIM]\n"
        f"Claim {target_claim_no}: {current_claim_text}\n\n"
        "[ALL PARENT CLAIMS]\n"
        f"{parent_context}"
    )

    return {
        "target_claim_no": target_claim_no,
        "parent_claims": parent_claims,
        "missing_parent_claims": missing_parent_claims,
        "current_claim_text": current_claim_text,
        "parent_context": parent_context,
        "all_parent_context": all_parent_context,
    }