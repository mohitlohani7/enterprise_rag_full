import pdfplumber
from typing import List

def load_pdf(path: str) -> str:
    """Load and return text from a PDF file."""
    text = ""
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            extracted = page.extract_text() or ""
            text += extracted + "\n"
    return text

def load_all_pdfs(folder_path: str) -> List[str]:
    """Load all PDFs in a folder and return list of (filename, text)."""
    import os
    items = []
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(".pdf"):
            full = os.path.join(folder_path, fname)
            items.append((fname, load_pdf(full)))
    return items
