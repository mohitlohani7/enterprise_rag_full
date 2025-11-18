import os

# -----------------------------
# MODEL ROUTING (Groq + OpenAI)
# -----------------------------
# Use only validated Groq models here.
# If you need other models, add only names supported by your Groq account.

GROQ_MODELS = {
    "default": {
        "provider": "groq",
        "model": "llama3.1-8b-instant",   # NEW VALID MODEL
        "key": os.getenv("GROQ_API_KEY"),
        "temperature": 0.1,
        "max_tokens": 400
    }
}

OPENAI_MODELS = {
    "default": {
        "provider": "openai",
        "model": "gpt-4o-mini",         # change if you prefer another valid OpenAI model
        "key": os.getenv("OPENAI_API_KEY"),
        "temperature": 0.2,
        "max_tokens": 400
    }
}


def select_model(query_type: str, prefer: str = "auto"):
    """
    Return a model configuration dict:
    {
        "provider": "groq" / "openai",
        "model": "...",
        "key": "...",
        "temperature": 0.1,
        "max_tokens": 400
    }
    prefer: "groq", "openai", or "auto"
    """
    # simple routing for now
    if prefer == "groq":
        return GROQ_MODELS["default"]
    if prefer == "openai":
        return OPENAI_MODELS["default"]

    # auto: prefer groq if key exists, else openai
    if GROQ_MODELS["default"].get("key"):
        return GROQ_MODELS["default"]
    if OPENAI_MODELS["default"].get("key"):
        return OPENAI_MODELS["default"]

    # fallback to groq config even if key is None (will error later with clear message)
    return GROQ_MODELS["default"]
