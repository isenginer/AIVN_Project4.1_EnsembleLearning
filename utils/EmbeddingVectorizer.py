"""
Class Embedding Vectorizer is the transfomer to convert the text to numpy array with library from SentenceTransfomer
The method of `SentenceTransformer`:
*
*
"""
from nltk.book import texts
import numpy as np
from streamlit import date_input
from transformers.models.tapas.tokenization_tapas import parse_text
from collections import Counter, defaultdict
from typing import List, Dict, Literal, Union

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sentence_transformers import SentenceTransformer

class EmbeddingVectorizer(object):
    def __init__(self,
                 model_name: str = "intfloat/multilingual-e5-base",
                 normalize: bool = True):
        self.model = SentenceTransformer(model_name)
        self.normalize = normalize

    def _format_input(self,
                      texts: List[str],
                      mode: Literal["query", "passage"] = "query") -> List[str]:
        if mode not in {"query", "passage"}:
            raise ValueError(f"Invalid mode {mode}")
        return [f"{mode}: {text.strip()}" for text in texts]

    def transform(self,
                  texts: List[str],
                  mode: Literal["query", "passage"] = "query") -> List[List[float]]:
        if mode == "raw":
            inputs = texts
        else:
            inputs = self._format_input(texts, mode)

        embeddings = self.model.encode(inputs, normalize_embeddings=self.normalize)
        return embeddings.tolist()

    def transform_numpy(self,
                        texts: List[str],
                        mode: Literal["query", "passage"] = "query") -> np.ndarray:
        return np.array(self.transform(texts, mode=mode))


def tfidf_vectorizer(texts: List[str]):
    pass

def bow_vectorizer(texts: List[str]):
    pass

def embedding_vectorizer(texts: List[str]):
    pass