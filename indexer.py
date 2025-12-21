"""Indexing utilities for TF-IDF matrices."""

from typing import Any, Dict, Iterable, Mapping, Optional

import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer


class Indexer:
    """Build TF-IDF vectorizers and matrices from tabular records.

    Attributes:
        records: DataFrame of input records to index.
        vectorizer_params: Parameters passed to `TfidfVectorizer`.
        matrices: Mapping of field name to TF-IDF matrix.
        vectorizers: Mapping of field name to fitted TF-IDF vectorizer.
    """
    def __init__(
        self,
        records: Iterable[Mapping[str, Any]],
        vectorizer_params: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Initialize an Indexer.

        Args:
            records: Iterable of record dicts to index.
            vectorizer_params: Parameters passed to `TfidfVectorizer`.
        """
        self.records = pd.DataFrame(records)
        self.vectorizer_params = vectorizer_params or {}
        self.matrices: Dict[str, spmatrix] = {}
        self.vectorizers: Dict[str, TfidfVectorizer] = {}

    def index_records(self) -> None:
        """Fit TF-IDF vectorizers and matrices for each record field."""
        for field in self.records.columns:
            vectorizer = TfidfVectorizer(**self.vectorizer_params)
            matrice = vectorizer.fit_transform(self.records[field])
            self.vectorizers[field] = vectorizer
            self.matrices[field] = matrice
