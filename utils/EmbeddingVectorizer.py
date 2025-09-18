"""
Class Embedding Vectorizer is the transformer to convert the text to numpy array with library from SentenceTransformer
The method of `SentenceTransformer`:
*
*
"""
from encodings import normalize_encoding

import numpy as np
import torch
from typing import List, Literal
from sentence_transformers import SentenceTransformer

class EmbeddingVectorizer(object):
    def __init__(self,
                 model_name: str = "intfloat/multilingual-e5-base",
                 normalize: bool = True,
                 device: str = "auto"):
        """
        Initialize the embedding vectorizer.
        Args:
        model_name: Name of the sentence transformer model
        normalize: Whether to normalize embeddings
        device: Device to run on ("auto", "cpu", "cuda", or specific device)
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.normalize = normalize
        self.model_name = model_name
        print(f"EmbeddingVectorizer initialized with model: {model_name} on device: {device}")

    def _format_input(self,
                      texts: List[str],
                      mode: Literal["query", "passage"] = "query") -> List[str]:
        if mode not in {"query", "passage", "raw"}:
            raise ValueError(f"Invalid mode {mode}. Must be 'query', 'passage' or 'raw'")

        if mode == "raw":
            return [text.strip() for text in texts]

        return [f"{mode}: {text.strip()}" for text in texts]

    def transform(self,
                  texts: List[str],
                  mode: Literal["query", "passage", "raw"] = "query",
                  batch_size: int=64,
                  show_progress_bar: bool = False) -> List[List[float]]:
        """
        Transform texts to embeddings.

        Args:
            texts: List of input texts
            mode: Processing mode ("query", "passage", or "raw")
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show progress bar

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Validate input type
        if not isinstance(texts, list):
            raise TypeError("texts must be a list")
        if not all((isinstance(text, str) for text in texts)):
            raise TypeError("all elements of texts must be a string")

        # format input base on mode

        inputs = self._format_input(texts, mode)

        try:
            embeddings = self.model.encode(
                inputs,
                normalize_embeddings=self.normalize,
            batch_size=batch_size,
                show_progress_bar=show_progress_bar,
            convert_to_tensor=False)
            return embeddings.tolist()
        except Exception as e:
            raise RuntimeError(f"Error during embeddings generation: {str(e)}")

    def transform_numpy(self,
                        texts: List[str],
                        mode: Literal["query", "passage", "raw"] = "query",
                        batch_size: int=64,
                        show_progress_bar: bool = False) -> np.ndarray:
        """
        Transform texts to embeddings and return as numpy array.
        Args:
            texts: List of input texts
            mode: Processing mode ("query", "passage", or "raw")
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show progress bar

        Returns:
            Numpy array of embeddings
        """
        return np.array(self.transform(texts, mode, batch_size, show_progress_bar))

    def encode_query(self, queries: List[str], ** kwargs) -> np.ndarray:
        """Convenience method for encoding queries."""
        return self.transform_numpy(queries, mode="query", **kwargs)

    def encode_passages(self, passages: List[str], **kwargs) -> np.ndarray:
        """Convenience method for encoding passages."""
        return self.transform_numpy(passages, mode="passage", **kwargs)

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        try:
            # Test with a dummy input to get dimension
            dummy_embedding = self.model.encode(["test"], convert_to_tensor=False)
            return dummy_embedding.shape[1]
        except Exception as e:
            raise RuntimeError(f"Could not determine embedding dimension: {str(e)}")

    def similarity(self,
                   embeddings1: np.ndarray,
                   embeddings2: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between two sets of embeddings.

        Args:
            embeddings1: First set of embeddings (N x D)
            embeddings2: Second set of embeddings (M x D)

        Returns:
            Similarity matrix (N x M)
        """
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(embeddings1, embeddings2)

    def __repr__(self) -> str:
        return f"EmbeddingVectorizer(model='{self.model_name}', device='{self.device}', normalize={self.normalize})"


def tfidf_vectorizer(X_train, X_test):
    """
    function to return the numpy array of train & test transformed from TfidfVectorizer
    :param X_train: train features
    :param X_test: test features
    :return: numpy array of train & test transformed from TfidfVectorizer
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    return np.array(X_train_tfidf.toarray()), np.array(X_test_tfidf.toarray())


def bow_vectorizer(X_train, X_test):
    """
    function to return the numpy array of train & test transformed from CountVectorizer
    :param X_train: train features
    :param X_test: test features
    :return: numpy array of train & test transformed from CountVectorizer
    """
    from sklearn.feature_extraction.text import CountVectorizer
    ctv = CountVectorizer()
    X_train_bow = ctv.fit_transform(X_train)
    X_test_bow = ctv.transform(X_test)
    return np.array(X_train_bow.toarray()), np.array(X_test_bow.toarray())


def embedding_vectorizer(X_train, X_test):
    """
    function to return the numpy array of train & test transformed from EmbeddingVectorizer
    :param X_train: train features
    :param X_test: test features
    :return: numpy array of train & test transformed from EmbeddingVectorizer
    """
    vectorizer = EmbeddingVectorizer(device="cpu")
    X_train_vt = vectorizer.transform_numpy(X_train)
    X_test_vt = vectorizer.transform_numpy(X_test)
    return X_train_vt, X_test_vt


def category_numerical(category: str=""):
    """
    function to return the value of categorical string in list
    :param category: the category string
    :return: value of categorical string in lis/t
    """
    category_list = ["astro-ph", "cond-mat", "cs", "math", "physics"]
    category_dict = dict(zip(category_list, range(1, 6)))
    if category not in category_list:
        return 0
    else:
        return category_dict[category]

if __name__ == "__main__":
    category = "math"
    print(category_numerical(category))