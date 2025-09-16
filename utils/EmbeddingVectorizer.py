"""
Class Embedding Vectorizer is the transformer to convert the text to numpy array with library from SentenceTransformer
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


def tfidf_vectorizer(X_train, X_test) -> numpy.ndarray:
    """
    function to return the numpy array of train & test transformed from TfidfVectorizer
    :param X_train: train features
    :param X_test: test features
    :return: numpy array of train & test transformed from TfidfVectorizer
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(tokenizer=lambda text: parse_text(text))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    return np.array(X_train_tfidf.toarray()), np.array(X_test_tfidf.toarray())

def bow_vectorizer(X_train, X_test) -> numpy.ndarray:
    """
    function to return the numpy array of train & test transformed from CountVectorizer
    :param X_train: train features
    :param X_test: test features
    :return: numpy array of train & test transformed from CountVectorizer
    """
    from sklearn.feature_extraction.text import CountVectorizer
    ctv = CountVectorizer(tokenizer=lambda text: parse_text(text))
    X_train_bow = ctv.fit_transform(X_train)
    X_test_bow = ctv.transform(X_test)
    return np.array(X_train_bow.toarray()), np.array(X_test_bow.toarray())

def embedding_vectorizer(X_train, X_test) -> numpy.ndarray:
    """
    function to return the numpy array of train & test transformed from EmbeddingVectorizer
    :param X_train: train features
    :param X_test: test features
    :return: numpy array of train & test transformed from EmbeddingVectorizer
    """
    vectorizer = EmbeddingVectorizer()
    X_train_vt = vectorizer.transform_numpy(X_train)
    X_test_vt = vectorizer.transform_numpy(X_test)
    return X_train_vt, X_test_vt

