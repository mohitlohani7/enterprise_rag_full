import re

def clean_text(text: str) -> str:
    """Basic text cleaning: normalize whitespace, remove weird chars, strip headers/footers heuristics."""
    # Remove repeated newlines and excessive spaces
    text = re.sub(r'\r', ' ', text)
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'\s{2,}', ' ', text)
    # Remove non-printable characters
    text = ''.join(ch for ch in text if ch.isprintable())
    # Optionally remove common page headers/footers heuristics (numbers, page words)
    text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Page \d+', '', text, flags=re.IGNORECASE)
    return text.strip()
