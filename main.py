"""Example entry point for building and querying a text search index."""

import json
from typing import Any, Dict, List

import requests

from text_search import TextSearch

DOCS_URL = (
    "https://raw.githubusercontent.com/"
    "alexeygrigorev/llm-rag-workshop/main/notebooks/documents.json"
)


def _get_documents() -> List[Dict[str, Any]]:
    """Load and flatten documents from the remote JSON source."""
    docs_response = requests.get(DOCS_URL, timeout=10)
    docs_response.raise_for_status()
    documents_raw = docs_response.json()

    documents = []

    for course in documents_raw:
        course_name = course['course']

        for doc in course['documents']:
            doc['course'] = course_name
            documents.append(doc)
    return documents


def main() -> None:
    """Run a sample search query against the documents."""
    documents = _get_documents()
    text_search = TextSearch(text_fields=['section', 'question', 'text'])
    text_search.fit(
        documents,
        vectorizer_params={'stop_words': 'english', 'min_df': 5},
    )
    raw_results = text_search.search(
        query='I just signed up. Is it too late to join the course?',
        n_results=5,
        boost={'question': 3.0},
        filters={'course': 'data-engineering-zoomcamp'})

    cleaned_results = json.dumps(raw_results, indent=4)
    print(cleaned_results)


if __name__ == '__main__':
    main()
