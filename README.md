# Enterprise RAG - Project Template

This repository contains a company-style RAG (Retrieval-Augmented Generation) prototype scaffold.
It is intentionally modular so you can plug-in real LLM providers such as Llama 3 (via key), Groq, OpenAI, or other APIs.

## Folder structure
```
enterprise_rag/
│── data/
│   └── pdfs/
│── src/
│   ├── loaders/
│   │   └── pdf_loader.py
│   ├── preprocess/
│   │   ├── clean_text.py
│   │   └── chunker.py
│   ├── retrievers/
│   │   ├── vector_retriever.py
│   │   ├── bm25_retriever.py
│   │   └── hybrid_retriever.py
│   ├── ranking/
│   │   └── reranker.py
│   ├── llm/
│   │   ├── query_classifier.py
│   │   ├── model_router.py
│   │   └── answer_validator.py
│   ├── pipeline/
│   │   └── rag_pipeline.py
│   └── utils/
│       └── logger.py
│── app.py
│── requirements.txt
│── README.md
```

## Quickstart (local)
1. Create virtual env and install requirements:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Place PDFs into `data/pdfs/`.

3. (Optional) Set environment variables for LLM providers in `.env`:
   ```
   LLAMA3_3B_KEY=your_llama3_key_here
   GROQ_API_KEY=your_groq_key_here
   OPENAI_API_KEY=your_openai_key_here
   ```

4. Run Streamlit:
   ```bash
   streamlit run app.py
   ```

## Where to add LLM provider logic
- `src/llm/model_router.py` returns a small dict describing model selection. Replace this logic with actual calls (OpenAI, Groq, or Llama).
- `src/pipeline/rag_pipeline.py` currently returns a simulated answer. Replace the placeholder with your LLM invocation (using selected provider) and include source citation.

## Notes
- `cross-encoder/ms-marco-MiniLM-L-6-v2` is used inside `src/ranking/reranker.py`. If GPU is available it's recommended.
- This scaffold focuses on production-like modules (hybrid retriever, reranker, validator, model routing). Add your API keys and provider-specific SDKs to integrate a real LLM.
