import os
import groq
from openai import OpenAI
from typing import List, Tuple

from src.llm.model_router import select_model
from src.llm.query_classifier import classify_query
from src.ranking.reranker import Reranker

# Optional: import groq exception class for fine handling
try:
    from groq.exceptions import GroqAPIError, BadRequestError
except Exception:
    # fallback names (SDK variants)
    GroqAPIError = Exception
    BadRequestError = Exception


class RAGPipeline:
    def __init__(self, retriever):
        """
        retriever: object with .search(query, k) -> list_of_chunks (strings)
        reranker: Reranker class that exposes rerank(query, candidates, top_k) -> [(chunk, score), ...]
        """
        self.retriever = retriever
        self.reranker = Reranker()
        self.groq_client = None

        # these can be set externally (from app.py)
        self.temperature = 0.1
        self.max_tokens = 400
        self.model_choice = "auto"  # "auto", "groq", or "openai"
        self.top_k = 5
        self.rerank = True

        # optional metadata set by app
        self.doc_map = None
        self.raw_chunks = None

    # -----------------------------
    # Load Groq client lazily
    # -----------------------------
    def load_groq(self, api_key: str):
        if self.groq_client is None:
            if not api_key:
                raise ValueError("Missing GROQ_API_KEY for Groq provider.")
            self.groq_client = groq.Groq(api_key=api_key)

    # -----------------------------
    # Safety helpers
    # -----------------------------
    def _clamp_temperature(self, t: float):
        try:
            t = float(t)
        except Exception:
            t = 0.1
        return max(0.0, min(2.0, t))

    def _clamp_max_tokens(self, m: int):
        try:
            m = int(m)
        except Exception:
            m = 400
        return max(1, min(8192, m))  # keep within reasonable limit

    # -----------------------------
    # Main ask function
    # -----------------------------
    def ask(self, query: str, k: int = 5) -> Tuple[str, List[Tuple[str, float]]]:
        """
        Returns (answer_text, reranked_list)
        reranked_list -> list of (chunk, score) pairs (may be empty)
        """

        # 1) classify
        q_type = classify_query(query)

        # 2) select model config
        model_conf = select_model(q_type, prefer=self.model_choice)
        provider = model_conf.get("provider", "groq")

        # 3) retrieve candidates
        try:
            candidates = self.retriever.search(query, k=max(k, 10))
        except Exception as e:
            # retrieval layer failed
            return f"Retrieval error: {e}", []

        # 4) optional reranking
        reranked = []
        try:
            if self.rerank:
                reranked = self.reranker.rerank(query, candidates, top_k=k)
            else:
                # if no reranker, create simple scored list (score 1.0)
                reranked = [(c, 1.0) for c in candidates[:k]]
        except Exception:
            # fail-safe
            reranked = [(c, 1.0) for c in candidates[:k]]

        # 5) build context from top chunks
        top_chunks = [c for c, *_ in reranked] if reranked else candidates[:k]
        context = "\n\n".join(top_chunks).strip()

        # safety: ensure non-empty context
        if not context:
            context = "No relevant context found in uploaded documents."

        # prepare final prompt (concise)
        prompt = (
            "You are an enterprise document assistant. Answer using ONLY the provided context. "
            "When you cite use square bracket style, e.g. [Source]. Do not hallucinate.\n\n"
            f"Query: {query}\n\nContext:\n{context}\n\nAnswer:"
        )

        # prepare model params
        temperature = self._clamp_temperature(model_conf.get("temperature", self.temperature))
        max_tokens = self._clamp_max_tokens(model_conf.get("max_tokens", self.max_tokens))

        # -------------------
        # GROQ provider
        # -------------------
        if provider == "groq":
            # try load groq client
            api_key = model_conf.get("key") or os.getenv("GROQ_API_KEY")
            try:
                self.load_groq(api_key)
            except Exception as e:
                return f"Groq initialization error: {e}", reranked

            try:
                completion = self.groq_client.chat.completions.create(
                    model=model_conf.get("model"),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                # newer SDK uses .choices[0].message.content
                answer = getattr(completion.choices[0].message, "content", None)
                if answer is None:
                    # try fallback keys
                    answer = completion.choices[0].message.get("content", "")
            except BadRequestError as be:
                # groq returned a 400-level error (bad model/params). Show actionable message.
                return f"Groq BadRequest: {str(be)} (check model name, temperature and max_tokens, and your GROQ_API_KEY)", reranked
            except GroqAPIError as ge:
                return f"Groq API error: {ge}", reranked
            except Exception as e:
                # generic network/SDK error fallback
                return f"Groq call failed: {e}", reranked

            return answer, reranked

        # -------------------
        # OPENAI provider
        # -------------------
        elif provider == "openai":
            api_key = model_conf.get("key") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                return "OpenAI API key missing in environment.", reranked
            try:
                client = OpenAI(api_key=api_key)
                completion = client.chat.completions.create(
                    model=model_conf.get("model"),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                answer = getattr(completion.choices[0].message, "content", None)
                if answer is None:
                    answer = completion.choices[0].message.get("content", "")
            except Exception as e:
                return f"OpenAI call failed: {e}", reranked

            return answer, reranked

        # -------------------
        # Unknown provider
        # -------------------
        else:
            return "No valid model provider configured.", reranked
