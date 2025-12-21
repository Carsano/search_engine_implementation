"""Text search utilities using TF-IDF and cosine similarity.

This module provides a lightweight in-memory text search helper backed by
TF-IDF vectorization and cosine similarity scoring.
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.sparse import spmatrix
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TextSearch:
    """In-memory text search over multiple fields.

    Attributes:
        text_fields: Iterable of field names to index for text search.
        matrices: Mapping of field name to TF-IDF matrix.
        vectorizers: Mapping of field name to fitted TF-IDF vectorizer.
    """
    def __init__(self, text_fields: Sequence[str]) -> None:
        """Initialize a TextSearch instance.

        Args:
            text_fields: Iterable of field names to index for text search.
        """
        self.text_fields = text_fields
        self.matrices: Dict[str, spmatrix] = {}
        self.vectorizers: Dict[str, TfidfVectorizer] = {}
        self.df: pd.DataFrame = pd.DataFrame()

    def _get_relevant_documents(
        self,
        n_results: int,
        score: NDArray[np.floating],
    ) -> pd.DataFrame:
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

    def _apply_mask(
        self,
        filters: Mapping[str, Any],
        score: NDArray[np.floating],
    ) -> NDArray[np.floating]:
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

    def _get_cosine_score(
        self,
        query: str,
        boost: Mapping[str, float],
        score: NDArray[np.floating],
    ) -> NDArray[np.floating]:
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

            field_boost = boost.get(field, 1.0)

            score = score + field_boost * f_score
        return score

    def fit(
        self,
        records: Iterable[Mapping[str, Any]],
        vectorizer_params: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Fit TF-IDF vectorizers and matrices from records.

        Args:
            records: Iterable of record dicts to index.
            vectorizer_params: Parameters passed to `TfidfVectorizer`.
        """
        self.df = pd.DataFrame(records)

        vectorizer_params = vectorizer_params or {}

        for field in self.text_fields:
            vectorizer = TfidfVectorizer(**vectorizer_params)
            matrice = vectorizer.fit_transform(self.df[field])
            self.vectorizers[field] = vectorizer
            self.matrices[field] = matrice

    def search(
        self,
        query: str,
        boost: Optional[Mapping[str, float]] = None,
        filters: Optional[Mapping[str, Any]] = None,
        n_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search the indexed records.

        Args:
            query: Query string to search for.
            boost: Mapping of field name to multiplicative boost.
            filters: Mapping of field name to exact value to keep.
            n_results: Number of results to return.

        Returns:
            List of matching records ordered by relevance.
        """
        boost = boost or {}
        filters = filters or {}

        score = np.zeros(len(self.df))
        score = self._get_cosine_score(query, boost, score)
        score = self._apply_mask(filters, score)
        relevant_documents = self._get_relevant_documents(n_results, score)
        return relevant_documents.to_dict(orient='records')
