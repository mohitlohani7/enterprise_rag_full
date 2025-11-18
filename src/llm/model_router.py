import os

# -----------------------------
# MODEL ROUTING (Groq + OpenAI)
# -----------------------------

VALID_GROQ_MODELS = [
    "gemma-7b-it",
    "mixtral-8x7b-32768",
    "llama3.1-70b-instant",
    "llama3.1-8b"
]

GROQ_MODELS = {
    "default": {
        "provider": "groq",
        "model": "gemma-7b-it",
        "key": os.getenv("GROQ_API_KEY"),
        "temperature": 0.1,
        "max_tokens": 400
    }
}

OPENAI_MODELS = {
    "default": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "key": os.getenv("OPENAI_API_KEY"),
        "temperature": 0.2,
        "max_tokens": 400
    }
}


def validate_groq_model(model_name: str) -> bool:
    return model_name in VALID_GROQ_MODELS


def groq_available() -> bool:
    conf = GROQ_MODELS["default"]
    return bool(conf["key"]) and validate_groq_model(conf["model"])


def openai_available() -> bool:
    return bool(OPENAI_MODELS["default"]["key"])


def select_model(query_type: str, prefer: str = "auto"):

    if prefer == "groq":
        if groq_available():
            return GROQ_MODELS["default"]
        return OPENAI_MODELS["default"]

    if prefer == "openai":
        if openai_available():
            return OPENAI_MODELS["default"]
        return GROQ_MODELS["default"]

    if groq_available():
        return GROQ_MODELS["default"]

    if openai_available():
        return OPENAI_MODELS["default"]

    return OPENAI_MODELS["default"]
