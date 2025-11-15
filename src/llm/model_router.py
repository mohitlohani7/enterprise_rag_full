import os

# -----------------------------
# MODEL ROUTING (Groq + OpenAI)
# -----------------------------

# VALID GROQ MODELS
GROQ_MODELS = {
    "default": {
        "provider": "groq",
        "model": "llama3-8b-8192",      # 100% valid model
        "key": os.getenv("GROQ_API_KEY"),
        "temperature": 0.1,
        "max_tokens": 400
    }
}

# VALID OPENAI MODELS
OPENAI_MODELS = {
    "default": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "key": os.getenv("OPENAI_API_KEY"),
        "temperature": 0.1,
        "max_tokens": 400
    }
}

# -----------------------------
# MODEL SELECTOR
# -----------------------------
def select_model(query_type: str):
    """
    Chooses the model based on query type.
    You can expand logic later.
    """
    # For now route everything to Groq (FASTER + FREE)
    return GROQ_MODELS["default"]
