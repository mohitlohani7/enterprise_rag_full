from src.llm.query_classifier import classify_query
from src.llm.model_router import select_model
from src.ranking.reranker import Reranker
from groq import Groq
from openai import OpenAI
import os


class RAGPipeline:
    def __init__(self, retriever):
        self.retriever = retriever
        self.reranker = Reranker()
        self.groq_client = None

    def load_groq(self, api_key):
        if self.groq_client is None:
            self.groq_client = Groq(api_key=api_key)

    def ask(self, query: str, k: int = 5):

        # classify query type
        q_type = classify_query(query)

        # choose model (OpenAI or Groq)
        model_conf = select_model(q_type)

        # retrieve candidates
        candidates = self.retriever.search(query, k=10)

        # rerank the chunks
        reranked = self.reranker.rerank(query, candidates, top_k=k)

        # join top chunks into final context
        top_chunks = [c[0] for c in reranked]
        context = "\n\n".join(top_chunks)

        provider = model_conf["provider"]

        # ------------------- GROQ LLaMA CALL -------------------
        if provider == "groq":
            self.load_groq(model_conf["key"])

            prompt = f"""
You are an enterprise document QA system.
Answer ONLY using the context provided.
Add citations like [Source].

Query: {query}

Context:
{context}

Answer:
"""

            completion = self.groq_client.chat.completions.create(
                model=model_conf["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=400
            )

            # NEW SDK FIX → message["content"] ❌  message.content ✔
            answer = completion.choices[0].message.content


        # ------------------- OPENAI CALL -------------------
        elif provider == "openai":
            client = OpenAI(api_key=model_conf["key"])

            prompt = f"""
You are a highly accurate document analysis AI.
Use ONLY the provided context to answer.
Do not hallucinate. Add citations.

Query: {query}

Context:
{context}

Answer:
"""

            completion = client.chat.completions.create(
                model=model_conf["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=400
            )

            # NEW SDK FIX HERE ALSO
            answer = completion.choices[0].message.content


        else:
            answer = "(No valid model provider found)"

        return answer, reranked
