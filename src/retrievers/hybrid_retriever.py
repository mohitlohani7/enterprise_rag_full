from .vector_retriever import VectorRetriever
from .bm25_retriever import BM25Retriever

class HybridRetriever:
    def __init__(self):
        self.vector = VectorRetriever()
        self.bm25 = BM25Retriever([])

    def index(self, chunks):
        self.vector.add_documents(chunks)
        self.bm25.index(chunks)

    def search(self, query, k=5):
        vec = self.vector.search(query, k)
        bm = self.bm25.search(query, k)

        combined = list(dict.fromkeys(vec + bm))
        return combined[:k]
