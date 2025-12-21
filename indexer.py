import pandas as pd
from typing import Any, Dict, Iterable, Mapping, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import spmatrix


class Indexer:
    def __init__(
            self,
            records: Iterable[Mapping[str, Any]],
            vectorizer_params: Optional[Mapping[str, Any]] = None
            ) -> None:
        self.records = pd.DataFrame(records)
        self.vectorizer_params = vectorizer_params
        self.matrices: Dict[str, spmatrix] = {}
        self.vectorizers: Dict[str, TfidfVectorizer] = {}

    def index_records(self):
        """Fit TF-IDF vectorizers and matrices from records.

        Args:
            records: Iterable of record dicts to index.
            vectorizer_params: Parameters passed to `TfidfVectorizer`.
        """
        for field in self.records.columns:
            vectorizer = TfidfVectorizer(**self.vectorizer_params)
            matrice = vectorizer.fit_transform(self.records[field])
            self.vectorizers[field] = vectorizer
            self.matrices[field] = matrice
