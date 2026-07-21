import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from enrich_patentmatch.claims.expand_paths import (
    build_dependency_graph,
    expand_paths,
    build_pathwise_contexts,
)


def main():
    claims = [
        {
            "claim_no": 1,
            "text": "A system comprising a processor.",
        },
        {
            "claim_no": 2,
            "text": (
                "The system according to claim 1, "
                "further comprising a memory."
            ),
        },
        {
            "claim_no": 3,
            "text": (
                "The system according to claim 1, "
                "further comprising a sensor."
            ),
        },
        {
            "claim_no": 4,
            "text": (
                "The system according to any one of "
                "claims 2 or 3, further comprising "
                "a transmitter."
            ),
        },
    ]

    graph = build_dependency_graph(claims)

    paths = expand_paths(
        graph=graph,
        claim_no=4,
    )

    contexts = build_pathwise_contexts(
        graph=graph,
        paths=paths,
        target_claim_no=4,
    )

    print("Expanded paths:")
    print(paths)

    print("\nPath-wise contexts:")

    for context in contexts:
        print("=" * 80)
        print("path_id:", context["path_id"])
        print(
            "path_claim_numbers:",
            context["path_claim_numbers"],
        )
        print("path_depth:", context["path_depth"])
        print(context["path_text"])

    assert paths == [
        [1, 2, 4],
        [1, 3, 4],
    ]

    assert len(contexts) == 2

    assert contexts[0]["path_id"] == 0
    assert contexts[0]["path_claim_numbers"] == [1, 2, 4]
    assert contexts[0]["path_depth"] == 3

    assert contexts[1]["path_id"] == 1
    assert contexts[1]["path_claim_numbers"] == [1, 3, 4]
    assert contexts[1]["path_depth"] == 3

    assert "Claim 1:" in contexts[0]["path_text"]
    assert "Claim 2:" in contexts[0]["path_text"]
    assert "Claim 4:" in contexts[0]["path_text"]

    print("\nAll path-wise context tests passed.")


if __name__ == "__main__":
    main()