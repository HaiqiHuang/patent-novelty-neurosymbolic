import re
from pathlib import Path

import epo_ops

from enrich_patentmatch.epo.client import get_epo_client


def publication_to_docdb(publication_number: str) -> epo_ops.models.Docdb:
    """
    Convert publication number like EP2873927A1 to EPO OPS Docdb object.

    EP2873927A1 ->
        country = EP
        doc_number = 2873927
        kind = A1
    """
    publication_number = publication_number.strip().upper()

    m = re.fullmatch(r"([A-Z]{2})(\d+)([A-Z]\d?)", publication_number)
    if not m:
        raise ValueError(f"Cannot parse publication number: {publication_number}")

    country, doc_number, kind = m.groups()

    return epo_ops.models.Docdb(
        doc_number,
        country,
        kind,
    )


def fetch_claims_xml(
    publication_number: str,
    cache_dir: str = "data/external/epo_claims",
    use_cache: bool = True,
) -> str:
    """
    Fetch claims XML from EPO OPS by publication number.

    Example:
        EP2873927A1 -> data/external/epo_claims/EP2873927A1.xml
    """
    publication_number = publication_number.strip().upper()

    cache_path = Path(cache_dir) / f"{publication_number}.xml"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if use_cache and cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    client = get_epo_client()
    docdb = publication_to_docdb(publication_number)

    response = client.published_data(
        reference_type="publication",
        input=docdb,
        endpoint="claims",
    )

    xml = response.text
    cache_path.write_text(xml, encoding="utf-8")

    return xml