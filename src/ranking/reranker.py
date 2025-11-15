from sentence_transformers import SentenceTransformer, util

class Reranker:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def rerank(self, query, candidates, top_k=5):
        passages = [c[0] for c in candidates]
        q_emb = self.model.encode(query)
        ctx_emb = self.model.encode(passages)

        scores = util.cos_sim(q_emb, ctx_emb)[0].tolist()

        combined = list(zip(passages, scores))
        combined = sorted(combined, key=lambda x: x[1], reverse=True)
        return combined[:top_k]
