from sentence_transformers import SentenceTransformer, util
from typing import List, Tuple
import torch

class VectorRetriever:
    def __init__(self, chunks: List[str], model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.chunks = chunks
        self.embeddings = self.model.encode(chunks, convert_to_tensor=True)

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        if len(self.chunks) == 0:
            return []

        q_emb = self.model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(q_emb, self.embeddings)[0]

        # SAFE-K prevent crash if k > available chunks
        k = min(k, len(self.chunks))

        # torch.topk fails if k > number of elements
        top_k_scores, top_k_indices = torch.topk(scores, k)

        results = []
        for score, idx in zip(top_k_scores, top_k_indices):
            results.append((self.chunks[int(idx)], float(score)))

        return results
