"""Text search utilities using TF-IDF and cosine similarity.

This module provides a lightweight in-memory text search helper backed by
TF-IDF vectorization and cosine similarity scoring.
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Any, Dict, List, Mapping, Optional, Sequence

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD, NMF
from indexer import Indexer


class TextSearch:
    """In-memory text search over multiple fields.

    Attributes:
        text_fields: Iterable of field names to index for text search.
        matrices: Mapping of field name to TF-IDF matrix.
        vectorizers: Mapping of field name to fitted TF-IDF vectorizer.
    """
    def __init__(
            self,
            text_fields: Sequence[str],
            indexer: Indexer,
            decomposer: TruncatedSVD | NMF = TruncatedSVD(n_components=16)
            ) -> None:
        """Initialize a TextSearch instance.

        Args:
            text_fields: Iterable of field names to index for text search.
        """
        self.text_fields = text_fields
        self.indexer = indexer
        self.decomposer = decomposer

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
        relevant_documents = self.indexer.records.iloc[indexes_relevant]
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
            mask = (self.indexer.records[field] == value).astype(int)
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
            matrice_emb, query_emb = self._get_embeddings(query, field)
            f_score = cosine_similarity(query_emb, matrice_emb).flatten()

            field_boost = boost.get(field, 1.0)

            score = score + field_boost * f_score
        return score

    def _get_embeddings(self, query, field):
        """
        Get embeddings for query and document matrix using SVD transformation.

        Vectorizes the input query using the appropriate field vectorizer,
        applies SVD decomposition to the document matrix, and transforms both
        the query and documents into the reduced embedding space.

        Args:
            query (str): The input query string to be embedded.
            field (str): The field name indicating which vectorizer
            and matrix to use.

        Returns:
            tuple: A tuple containing:
                - matrice_emb (ndarray): SVD-transformed document
                matrix embeddings.
                - query_emb (ndarray): SVD-transformed query embedding.
        """
        query_vectorized = (
                self.indexer.vectorizers[field].transform([query])
            )
        matrice = self.indexer.matrices[field]
        matrice_emb = self.decomposer.fit_transform(matrice)
        query_emb = self.decomposer.transform(query_vectorized)
        return matrice_emb, query_emb

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

        score = np.zeros(len(self.indexer.records))
        score = self._get_cosine_score(query, boost, score)
        score = self._apply_mask(filters, score)
        relevant_documents = self._get_relevant_documents(n_results, score)
        return relevant_documents.to_dict(orient='records')
