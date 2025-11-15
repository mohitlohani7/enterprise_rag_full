class HybridRetriever:
    def __init__(self, bm25, vector):
        self.bm25 = bm25
        self.vector = vector

    def search(self, query, k=5):
        # safe k
        total_chunks = len(self.bm25.chunks)
        if total_chunks == 0:
            return []

        k = min(k, total_chunks)

        bm25_results = self.bm25.search(query, k)
        vec_results = self.vector.search(query, k)

        combined = bm25_results + vec_results
        combined = sorted(combined, key=lambda x: x[1], reverse=True)

        # dedupe
        seen = set()
        unique = []
        for chunk, score in combined:
            if chunk not in seen:
                seen.add(chunk)
                unique.append((chunk, score))
                if len(unique) >= k:
                    break
        return unique
