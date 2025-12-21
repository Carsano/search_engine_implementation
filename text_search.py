"""Text search utilities using TF-IDF and cosine similarity.

This module provides a lightweight in-memory text search helper backed by
TF-IDF vectorization and cosine similarity scoring.
"""

import pandas as pd
import numpy as np
from typing import Array, Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TextSearch:
    """In-memory text search over multiple fields.

    Attributes:
        text_fields: Iterable of field names to index for text search.
        matrices: Mapping of field name to TF-IDF matrix.
        vectorizers: Mapping of field name to fitted TF-IDF vectorizer.
    """
    def __init__(self, text_fields: Array = []) -> None:
        """Initialize a TextSearch instance.

        Args:
            text_fields: Iterable of field names to index for text search.
        """
        self.text_fields = text_fields
        self.matrices: Dict = {}
        self.vectorizers: Dict = {}

    def _get_relevant_documents(self, n_results, score):
        """Return the top-N documents according to the score vector.

        Args:
            n_results: Number of results to return.
            score: Array-like scores aligned with the DataFrame.

        Returns:
            DataFrame containing the most relevant documents.
        """
        indexes_relevant = np.argsort(score).tail(n_results)
        relevant_documents = self.df.iloc[indexes_relevant]
        return relevant_documents

    def _apply_mask(self, filters, score):
        """Apply exact-match filters to the score vector.

        Args:
            filters: Mapping of field name to exact value to keep.
            score: Array-like scores aligned with the DataFrame.

        Returns:
            Updated score array after applying the filters.
        """
        for field, value in filters.items():
            mask = (self.df[field] == value).astype(int)
            score = score * mask
        return score

    def _get_cosine_score(self, query, boost, score):
        """Compute cosine similarity scores for the query.

        Args:
            query: Query string to search for.
            boost: Mapping of field name to multiplicative boost.
            score: Array-like scores to update.

        Returns:
            Updated score array after adding per-field similarities.
        """
        for field in self.text_fields:
            query_vectorized = self.vectorizers[field].transform([query])
            matrice = self.matrices[field]
            f_score = cosine_similarity(matrice, query_vectorized)

            boost = boost.get(field, 1.0)

            score = score + boost * f_score
        return score

    def fit(self, records, vectorizer_params: Dict = {}) -> None:
        """Fit TF-IDF vectorizers and matrices from records.

        Args:
            records: Iterable of record dicts to index.
            vectorizer_params: Parameters passed to `TfidfVectorizer`.
        """
        self.df = pd.DataFrame(records)

        for field in self.text_fields:
            vectorizer = TfidfVectorizer(**vectorizer_params)
            matrice = vectorizer.fit_transform(self.df[field])
            self.vectorizers[field] = vectorizer
            self.matrices[field] = matrice

    def search(
            self,
            query: str,
            boost: Dict = {},
            filters: Dict = {},
            n_results: int = 10):
        """Search the indexed records.

        Args:
            query: Query string to search for.
            boost: Mapping of field name to multiplicative boost.
            filters: Mapping of field name to exact value to keep.
            n_results: Number of results to return.

        Returns:
            List of matching records ordered by relevance.
        """
        score = np.zeros(len(self.df))
        score = self._get_cosine_score(query, boost, score)
        score = self._apply_mask(filters, score)
        relevant_documents = self._get_relevant_documents(n_results, score)
        return relevant_documents.to_dict(orient='records')
