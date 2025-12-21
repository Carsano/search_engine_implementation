"""
Module pour de la recherche textuelle
"""

import pandas as pd
import numpy as np
from typing import Array, Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TextSearch:
    """
    Docstring pour TextSearch
    """
    def __init__(self, text_fields: Array = []) -> None:
        self.text_fields = text_fields
        self.matrices: Dict = {}
        self.vectorizers: Dict = {}

    def fit(self, records, vectorizer_params: Dict = {}) -> None:
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
        score = np.zeros(len(self.df))

        for field in self.text_fields:
            query_vectorized = self.vectorizers[field].transform([query])
            matrice = self.matrices[field]
            f_score = cosine_similarity(matrice, query_vectorized)

            boost = boost.get(field, 1.0)

            score = score + boost * f_score

        for field, value in filters.items():
            mask = (self.df[field] == value).astype(int)
            score = score * mask

        indexes_relevant = np.argsort(score).tail(n_results)
        relevant_documents = self.df.iloc[indexes_relevant]
        return relevant_documents.to_dict(orient='records')
