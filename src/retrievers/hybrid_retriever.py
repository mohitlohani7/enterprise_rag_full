from .vector_retriever import VectorRetriever
from .bm25_retriever import BM25Retriever


class HybridRetriever:
    def __init__(self):
        self.vector = VectorRetriever()
        self.bm25 = None      # DO NOT initialize yet

    def index(self, chunks):
        # Index Chroma
        self.vector.add_documents(chunks)

        # Index BM25 only when chunks exist
        if chunks and len(chunks) > 0:
            self.bm25 = BM25Retriever(chunks)

    def search(self, query, k=5):
        # -------- Vector Search --------
        vec_results = self.vector.search(query, k)

        # -------- BM25 Search --------
        if self.bm25:
            bm25_results = self.bm25.search(query, k)
        else:
            bm25_results = []

        # -------- Combine (Unique) --------
        combined = list(dict.fromkeys(vec_results + bm25_results))
        return combined[:k]
