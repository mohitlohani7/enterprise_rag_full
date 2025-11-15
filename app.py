# app.py -- Enterprise Chat UI for RAG (Evidence toggle fixed)
import streamlit as st
import os
from dotenv import load_dotenv

from src.loaders.pdf_loader import load_all_pdfs
from src.preprocess.clean_text import clean_text
from src.preprocess.chunker import chunk_text

from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.vector_retriever import VectorRetriever
from src.retrievers.hybrid_retriever import HybridRetriever

from src.pipeline.rag_pipeline import RAGPipeline
from src.utils.logger import get_logger
from datetime import datetime

load_dotenv()
logger = get_logger("enterprise_rag")

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- Session init ----------------
def ensure_session():
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "pipeline" not in st.session_state:
        st.session_state["pipeline"] = None
    if "docs_indexed" not in st.session_state:
        st.session_state["docs_indexed"] = []
    if "last_index_time" not in st.session_state:
        st.session_state["last_index_time"] = None

ensure_session()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("## ⚙️ Controls")

    uploaded_files = st.file_uploader(
        "Upload PDF files (or drag & drop)", type=["pdf"], accept_multiple_files=True
    )

    st.markdown("### Model & Generation")
    model_choice = st.selectbox(
        "Model routing", ["auto (default)", "groq (Llama)", "openai (GPT)"], index=0
    )

    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)
    max_tokens = st.slider("Max tokens", 100, 2000, 600, 50)

    st.markdown("### Retrieval")
    top_k = st.slider("Chunks to retrieve (k)", 3, 12, 5)
    rerank = st.checkbox("Enable reranker", value=True)

    st.markdown("### UI")
    theme = st.radio("Theme", ["Professional Dark", "Professional Light"])
    show_chunk_scores = st.checkbox("Show evidence", value=False)  # default OFF

    clear_chat_btn = st.button("Clear chat history")

# Clear chat
if clear_chat_btn:
    st.session_state["chat_history"] = []
    st.success("Chat cleared.")

# ---------------- Theme ----------------
if theme == "Professional Dark":
    bg = "#0b1220"
    card = "#131722"
    text_color = "#e6eef6"
else:
    bg = "#ffffff"
    card = "#f7f9fb"
    text_color = "#0b1220"

# ---------------- Header ----------------
st.markdown(
    f"""
    <div style="background: linear-gradient(90deg,#020024,#090979,#00d4ff);
                padding:18px;border-radius:10px;margin-bottom:8px">
        <h2 style="color:white;margin:0">🧠 Enterprise Document AI — Chat Mode</h2>
        <div style="color:#dfeeff;font-size:14px;margin-top:6px">
          Upload PDFs, ask questions, and get evidence-backed answers (optional).
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- Layout ----------------
left_col, right_col = st.columns([1, 2])

# ---------------- LEFT COLUMN ----------------
with left_col:
    st.markdown("### 📚 Document Explorer")
    pdf_folder = "data/pdfs"
    os.makedirs(pdf_folder, exist_ok=True)

    # save files
    if uploaded_files:
        for up in uploaded_files:
            path = os.path.join(pdf_folder, up.name)
            with open(path, "wb") as f:
                f.write(up.read())
        st.success("Files uploaded!")

    # show files
    files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]
    if files:
        st.markdown("**Indexed files:**")
        for f in files:
            st.markdown(f"- {f}")
    else:
        st.info("No PDFs uploaded.")

    st.markdown("---")
    st.markdown("### 🛠 Index / Re-index")

    if st.button("Load & Index PDFs"):
        items = load_all_pdfs(pdf_folder)
        all_chunks = []
        doc_map = []

        for fname, txt in items:
            cleaned = clean_text(txt)
            chunks = chunk_text(cleaned)
            all_chunks.extend(chunks)
            doc_map.extend([(fname, i) for i in range(len(chunks))])

        # retrievers
        bm25 = BM25Retriever(all_chunks)
        vector = VectorRetriever(all_chunks)
        hybrid = HybridRetriever(bm25, vector)

        pipeline = RAGPipeline(hybrid)
        pipeline.doc_map = doc_map
        pipeline.raw_chunks = all_chunks

        pipeline.temperature = temperature
        pipeline.max_tokens = max_tokens
        pipeline.model_choice = model_choice
        pipeline.top_k = top_k
        pipeline.rerank = rerank

        st.session_state["pipeline"] = pipeline
        st.session_state["last_index_time"] = datetime.now().isoformat()

        st.success(f"Indexed {len(files)} files!")

    st.markdown("---")
    if st.session_state["last_index_time"]:
        st.write("Indexed at:", st.session_state["last_index_time"])

# ---------------- RIGHT COLUMN ----------------
with right_col:
    st.markdown("### 💬 Chat with your documents")

    prompt = st.text_area("Ask anything", height=80)

    col1, col2, col3 = st.columns([1, 1, 1])
    ask_btn = col1.button("Ask")
    regen_btn = col2.button("Regenerate")
    export_btn = col3.button("Export chat")

    pipeline = st.session_state.get("pipeline", None)

    if not pipeline:
        st.info("Index PDFs first.")
    else:
        # update generation params
        pipeline.temperature = temperature
        pipeline.max_tokens = max_tokens
        pipeline.model_choice = model_choice
        pipeline.top_k = top_k
        pipeline.rerank = rerank

        if ask_btn and prompt.strip():
            st.session_state["chat_history"].append(
                {"role": "user", "text": prompt, "meta": {}}
            )

            with st.spinner("Thinking..."):
                answer, ranked = pipeline.ask(prompt, k=top_k)

            st.session_state["chat_history"].append(
                {"role": "assistant", "text": answer, "meta": {"ranked": ranked}}
            )

        if regen_btn:
            last = None
            for msg in reversed(st.session_state["chat_history"]):
                if msg["role"] == "user":
                    last = msg["text"]
                    break
            if last:
                with st.spinner("Regenerating..."):
                    answer, ranked = pipeline.ask(last, k=top_k)
                    st.session_state["chat_history"].append(
                        {"role": "assistant", "text": answer, "meta": {"ranked": ranked}}
                    )

        # ---------------- CHAT HISTORY UI ----------------
        for msg in st.session_state["chat_history"]:
            if msg["role"] == "user":
                st.markdown(
                    f"<div style='background:{card};padding:12px;border-radius:8px;margin:6px 0;'>"
                    f"<b style='color:#7dd3fc;'>You:</b> {msg['text']}</div>",
                    unsafe_allow_html=True,
                )
            else:
                ranked = msg.get("meta", {}).get("ranked", [])
                assistant_html = (
                    f"<div style='background:#0b1220;color:white;padding:12px;border-radius:8px;margin:6px 0;'>"
                    f"<b>AI:</b><br><div style='margin-top:6px'>{msg['text']}</div>"
                )

                # ---------------- FIXED: Evidence toggle ----------------
                if show_chunk_scores and ranked:
                    assistant_html += "<hr style='border:none;height:1px;background:#222;margin:8px 0;'>"
                    assistant_html += "<div style='font-size:13px;color:#cbd5e1'>Top Evidence:</div>"
                    for idx, (chunk, score) in enumerate(ranked):
                        snippet = chunk.replace("\n", " ")[:300]
                        assistant_html += f"<div style='padding:6px;margin:6px 0;background:#111;border-radius:6px;'>"
                        assistant_html += f"<b>Rank {idx+1} | Score: {round(score,4)}</b>"
                        assistant_html += f"<div style='margin-top:6px'>{snippet}...</div></div>"

                assistant_html += "</div>"
                st.markdown(assistant_html, unsafe_allow_html=True)

        if export_btn:
            with open("chat_export.txt", "w", encoding="utf-8") as f:
                for m in st.session_state["chat_history"]:
                    f.write(f"{m['role'].upper()}: {m['text']}\n\n")
            st.success("Exported to chat_export.txt")

st.markdown("---")
st.caption("Enterprise RAG — free edition.")
