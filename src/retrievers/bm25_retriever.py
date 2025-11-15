from rank_bm25 import BM25Okapi
from typing import List, Tuple

class BM25Retriever:
    def __init__(self, chunks: List[str]):
        self.chunks = chunks
        tokenized = [c.split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        if len(self.chunks) == 0:
            return []

        scores = self.bm25.get_scores(query.split())

        # safe k handling
        k = min(k, len(scores))

        top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        return [(self.chunks[i], float(scores[i])) for i in top_k]
