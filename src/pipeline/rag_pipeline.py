from src.llm.query_classifier import classify_query
from src.llm.model_router import select_model
from src.ranking.reranker import Reranker

from groq import Groq
from openai import OpenAI


class RAGPipeline:
    def __init__(self, retriever):
        self.retriever = retriever
        self.reranker = Reranker()
        self.temperature = 0.1
        self.max_tokens = 400
        self.model_choice = "auto"
        self.top_k = 5
        self.rerank = True


    def _safe_groq(self, model_conf, prompt):
        try:
            client = Groq(api_key=model_conf["key"])
            out = client.chat.completions.create(
                model=model_conf["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return out.choices[0].message.content
        except Exception as e:
            print("❌ GROQ FAILED:", e)
            return None


    def _safe_openai(self, model_conf, prompt):
        try:
            client = OpenAI(api_key=model_conf["key"])
            out = client.chat.completions.create(
                model=model_conf["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return out.choices[0].message.content
        except Exception as e:
            print("❌ OPENAI FAILED:", e)
            return None


    def ask(self, query: str, k: int = 5):
        q_type = classify_query(query)
        model_conf = select_model(q_type, prefer=self.model_choice)

        candidates = self.retriever.search(query, k)
        reranked = self.reranker.rerank(query, candidates, top_k=k)

        context = "\n\n".join([c[0] for c in reranked])

        prompt = f"""
Use ONLY the following context to answer.
If missing, say: "Not available in documents."

Question:
{query}

Context:
{context}

Answer:
"""

        answer = None

        if model_conf["provider"] == "groq":
            answer = self._safe_groq(model_conf, prompt)
            if answer is None:
                fallback = select_model(q_type, "openai")
                answer = self._safe_openai(fallback, prompt)

        else:
            answer = self._safe_openai(model_conf, prompt)
            if answer is None:
                fallback = select_model(q_type, "groq")
                answer = self._safe_groq(fallback, prompt)

        if answer is None:
            answer = "⚠ LLM call failed. Check your GROQ_API_KEY and OPENAI_API_KEY."

        return answer, reranked
