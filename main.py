"""Example entry point for building and querying a text search index."""

import json
import argparse
from typing import Any, Dict, List, Mapping, Optional

import requests

from text_search import TextSearch
from indexer import Indexer

DEFAULT_DOCS_URL = (
    "https://raw.githubusercontent.com/"
    "alexeygrigorev/llm-rag-workshop/main/notebooks/documents.json"
)
DEFAULT_QUERY = None
DEFAULT_TOP_RESULTS = 5
DEFAULT_BOOST = {"question": 3.0}
DEFAULT_FILTERS = {"course": "data-engineering-zoomcamp"}


def _get_documents(docs_url: str) -> List[Dict[str, Any]]:
    """Load and flatten documents from the remote JSON source."""
    docs_response = requests.get(docs_url, timeout=10)
    docs_response.raise_for_status()
    documents_raw = docs_response.json()

    documents = []

    for course in documents_raw:
        course_name = course['course']

        for doc in course['documents']:
            doc['course'] = course_name
            documents.append(doc)
    return documents


def _parse_optional_json(value: Optional[str]) -> Mapping[str, Any]:
    """Parse a JSON object argument or return an empty mapping."""
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {value}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object for mapping arguments.")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--docs-url",
        default=DEFAULT_DOCS_URL,
        help="URL to the JSON documents source.",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Search query string.",
    )
    parser.add_argument(
        "--top-results",
        type=int,
        default=DEFAULT_TOP_RESULTS,
        help="Number of results to return.",
    )
    parser.add_argument(
        "--boost",
        default=json.dumps(DEFAULT_BOOST),
        help="JSON object mapping field names to boost values.",
    )
    parser.add_argument(
        "--filters",
        default=json.dumps(DEFAULT_FILTERS),
        help="JSON object mapping field names to exact-match filters.",
    )
    return parser


def main() -> None:
    """Run a sample search query against the documents."""
    parser = _build_parser()
    args = parser.parse_args()

    documents = _get_documents(args.docs_url)
    indexer = Indexer(
        records=documents,
        vectorizer_params={'stop_words': 'english', 'min_df': 5})
    indexer.index_records()
    text_search = TextSearch(
        text_fields=['section', 'question', 'text'],
        indexer=indexer)

    raw_results = text_search.search(
        query=args.query,
        n_results=args.top_results,
        boost=_parse_optional_json(args.boost),
        filters=_parse_optional_json(args.filters)
    )

    cleaned_results = json.dumps(raw_results, indent=4)
    print(cleaned_results)


if __name__ == '__main__':
    main()
