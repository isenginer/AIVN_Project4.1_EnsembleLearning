"""
Class Embedding Vectorizer is the transfomer to convert the text to numpy array with library from SentenceTransfomer
The method of `SentenceTransformer`:
*
*
"""
from sentence_transformers import SentenceTransformer


class EmbeddingVectorizer(object):
    def __init__(self, model=SentenceTransformer, ):
        pass