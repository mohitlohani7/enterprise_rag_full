import os
import groq
from openai import OpenAI

from src.llm.model_router import select_model
from src.llm.query_classifier import classify_query
from src.ranking.reranker import Reranker


class RAGPipeline:
    def __init__(self, retriever):
        self.retriever = retriever
        self.reranker = Reranker()
        self.groq_client = None

    # -----------------------------
    # Load Groq client
    # -----------------------------
    def load_groq(self, api_key):
        if self.groq_client is None:
            self.groq_client = groq.Groq(api_key=api_key)

    # -----------------------------
    # Main ASK
    # -----------------------------
    def ask(self, query: str, k: int = 5):

        # 1. classify
        q_type = classify_query(query)

        # 2. choose model
        model_conf = select_model(q_type)
        provider = model_conf["provider"]

        # 3. retrieve chunks
        candidates = self.retriever.search(query, k=10)

        # 4. rerank
        reranked = self.reranker.rerank(query, candidates, top_k=k)

        # 5. merge context
        top_chunks = [c[0] for c in reranked]
        context = "\n\n".join(top_chunks).strip()

        # SAFETY: context empty → avoid Groq BadRequest
        if not context:
            context = "No relevant context was found in uploaded documents."

        # FINAL PROMPT
        prompt = f"""
You are an enterprise-grade RAG document assistant.
Answer using ONLY the given context.
Always cite evidence using [Source].

Query:
{query}

Context:
{context}

Answer:
"""

        # -------------------------------------
        #     GROQ CALL (LATEST SDK)
        # -------------------------------------
        if provider == "groq":
            self.load_groq(model_conf["key"])

            completion = self.groq_client.chat.completions.create(
                model=model_conf["model"],                # VALID MODEL
                messages=[{"role": "user", "content": prompt}],
                temperature=model_conf["temperature"],
                max_tokens=model_conf["max_tokens"]
            )

            return completion.choices[0].message.content, reranked

        # -------------------------------------
        #     OPENAI CALL
        # -------------------------------------
        elif provider == "openai":
            client = OpenAI(api_key=model_conf["key"])

            completion = client.chat.completions.create(
                model=model_conf["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=model_conf["temperature"],
                max_tokens=model_conf["max_tokens"]
            )

            return completion.choices[0].message.content, reranked

        else:
            return "(No valid provider found)", reranked
