from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from enrich_patentmatch.epo.parse_ops_xml import parse_claims_from_xml
from enrich_patentmatch.claims.expand_paths import (
    build_dependency_graph,
    expand_paths,
    render_claim_path,
)


def main():
    xml_path = PROJECT_ROOT / "tests" / "fixtures" / "sample_claims_EP3045974A1.xml"
    xml = xml_path.read_text(encoding="utf-8")
    claims = parse_claims_from_xml(xml)

    graph = build_dependency_graph(claims)

    claim_no = 14
    paths = expand_paths(graph, claim_no)

    print(f"Claim {claim_no} has {len(paths)} expanded paths.")
    print()

    for path in paths[:len(paths)]:
        print(path)
        print(render_claim_path(graph, path))
        print("-" * 80)

    assert all(path[-1] == 14 for path in paths)
    assert [1, 14] in paths
    assert [1, 2, 14] in paths
    assert [1, 2, 3, 14] in paths

    print("Path expansion test passed.")


if __name__ == "__main__":
    main()