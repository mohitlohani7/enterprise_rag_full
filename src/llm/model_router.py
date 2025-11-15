import os

def select_model(query_type: str) -> dict:
    """
    Smart router: route between Groq (Llama3) and OpenAI (GPT-4o).
    No UI change required.
    """

    groq_key = os.getenv("GROQ_API_KEY")
    llama_model = os.getenv("LLAMA_MODEL", "llama3-8b-8192")

    openai_key = os.getenv("OPENAI_API_KEY")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Simple but effective routing rules
    if query_type in ("definition", "question"):
        return {
            "provider": "groq",
            "model": llama_model,
            "key": groq_key
        }

    # Complex → OpenAI (better reasoning)
    if query_type in ("summarization", "analysis", "comparison"):
        return {
            "provider": "openai",
            "model": openai_model,
            "key": openai_key
        }

    # Default
    return {
        "provider": "groq",
        "model": llama_model,
        "key": groq_key
    }
