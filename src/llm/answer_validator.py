from sentence_transformers import SentenceTransformer, util

class AnswerValidator:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def validate(self, answer: str, contexts: list) -> dict:
        # compute similarity between answer and contexts
        emb_ans = self.model.encode(answer, convert_to_tensor=True)
        emb_ctx = self.model.encode(contexts, convert_to_tensor=True)
        sims = util.cos_sim(emb_ans, emb_ctx)[0].tolist()
        return {'max_similarity': max(sims) if sims else 0.0, 'avg_similarity': sum(sims)/len(sims) if sims else 0.0}
