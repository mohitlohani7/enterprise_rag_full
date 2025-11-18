# app.py — Enterprise RAG with ChromaDB (FAISS REMOVED)
import streamlit as st
import os
from dotenv import load_dotenv
from datetime import datetime

# Load modules
from src.loaders.pdf_loader import load_all_pdfs
from src.preprocess.clean_text import clean_text
from src.preprocess.chunker import chunk_text

from src.retrievers.hybrid_retriever import HybridRetriever
from src.pipeline.rag_pipeline import RAGPipeline
from src.utils.logger import get_logger

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

    if "last_index_time" not in st.session_state:
        st.session_state["last_index_time"] = None


ensure_session()


# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("## ⚙️ Controls")

    uploaded_files = st.file_uploader(
        "Upload PDF files", type=["pdf"], accept_multiple_files=True
    )

    st.markdown("### Model & Generation")
    model_choice = st.selectbox(
        "Model routing", ["auto (default)", "groq (Llama)", "openai (GPT)"], index=0
    )

    temperature = st.slider("Temperature", 0.0, 1.0, 0.1)
    max_tokens = st.slider("Max tokens", 100, 2000, 600)

    st.markdown("### Retrieval")
    top_k = st.slider("Chunks (k)", 3, 12, 5)
    rerank = st.checkbox("Enable reranker", value=True)

    st.markdown("### UI Options")
    theme = st.radio("Theme", ["Professional Dark", "Professional Light"])
    show_chunk_scores = st.checkbox("Show evidence", value=False)

    if st.button("Clear Chat"):
        st.session_state["chat_history"] = []
        st.success("Chat cleared!")


# ---------------- THEME ----------------
if theme == "Professional Dark":
    card = "#131722"
else:
    card = "#f7f9fb"


# ---------------- Header ----------------
st.markdown(
    """
    <div style="background:linear-gradient(90deg,#020024,#090979,#00d4ff);
                padding:18px;border-radius:10px;margin-bottom:12px;">
      <h2 style="color:white;margin:0">🧠 Enterprise Document AI — Chat Mode</h2>
      <div style="color:#dfeeff;font-size:14px;margin-top:6px">
        Upload PDFs, ask questions, and get context-grounded answers.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------- Layout ----------------
left, right = st.columns([1, 2])


# ---------------- LEFT COLUMN ----------------
with left:
    st.markdown("### 📚 Document Explorer")

    pdf_dir = "data/pdfs"
    os.makedirs(pdf_dir, exist_ok=True)

    # Save files
    if uploaded_files:
        for file in uploaded_files:
            with open(os.path.join(pdf_dir, file.name), "wb") as f:
                f.write(file.read())
        st.success("Files uploaded!")

    # List existing files
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    if pdf_files:
        st.write("**Indexed files:**")
        for f in pdf_files:
            st.markdown(f"- {f}")
    else:
        st.info("No PDFs uploaded.")

    st.markdown("---")
    st.markdown("### 🛠 Index / Re-index")

    if st.button("Load & Index PDFs"):
        raw_docs = load_all_pdfs(pdf_dir)

        all_chunks = []
        doc_map = []

        for fname, text in raw_docs:
            cleaned = clean_text(text)
            chunks = chunk_text(cleaned)
            all_chunks.extend(chunks)
            doc_map.extend([(fname, i) for i in range(len(chunks))])

        # Hybrid retriever (CHROMA + BM25)
        hybrid = HybridRetriever()
        hybrid.index(all_chunks)

        pipeline = RAGPipeline(hybrid)
        pipeline.doc_map = doc_map
        pipeline.raw_chunks = all_chunks

        st.session_state["pipeline"] = pipeline
        st.session_state["last_index_time"] = datetime.now().isoformat()

        st.success(f"Indexed {len(pdf_files)} files successfully!")

    st.markdown("---")
    if st.session_state["last_index_time"]:
        st.caption("Indexed at: " + st.session_state["last_index_time"])


# ---------------- RIGHT COLUMN ----------------
with right:
    st.markdown("### 💬 Ask your documents")

    query = st.text_area("Ask anything", height=80)

    col1, col2, col3 = st.columns(3)
    ask_btn = col1.button("Ask")
    regen_btn = col2.button("Regenerate")
    export_btn = col3.button("Export")

    pipeline = st.session_state["pipeline"]

    if not pipeline:
        st.info("Index PDFs first.")
    else:
        # update settings
        pipeline.temperature = temperature
        pipeline.max_tokens = max_tokens
        pipeline.model_choice = model_choice
        pipeline.top_k = top_k
        pipeline.rerank = rerank

        if ask_btn and query.strip():
            st.session_state["chat_history"].append({"role": "user", "text": query})
            with st.spinner("Thinking..."):
                answer, ranked = pipeline.ask(query, k=top_k)
            st.session_state["chat_history"].append(
                {"role": "assistant", "text": answer, "ranked": ranked}
            )

        if regen_btn:
            # regenerate last user query
            last_q = None
            for m in reversed(st.session_state["chat_history"]):
                if m["role"] == "user":
                    last_q = m["text"]
                    break
            if last_q:
                with st.spinner("Regenerating..."):
                    answer, ranked = pipeline.ask(last_q, k=top_k)
                st.session_state["chat_history"].append(
                    {"role": "assistant", "text": answer, "ranked": ranked}
                )

        # render chat
        for msg in st.session_state["chat_history"]:
            if msg["role"] == "user":
                st.markdown(
                    f"""
                    <div style="background:{card};padding:12px;border-radius:8px;margin:6px 0;">
                        <b style="color:#7dd3fc;">You:</b> {msg['text']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                html = f"""
                <div style="background:#0b1220;color:white;padding:12px;border-radius:8px;margin:6px 0;">
                    <b>AI:</b><br> {msg['text']}
                """

                if show_chunk_scores:
                    html += "<hr><div style='font-size:13px;'>Evidence:</div>"
                    for i, (chunk, score) in enumerate(msg.get("ranked", [])):
                        snippet = chunk.replace("\n", " ")[:250]
                        html += f"""
                        <div style="background:#111;padding:6px;margin:6px 0;border-radius:6px;">
                            <b>Rank {i+1} | Score: {round(score,4)}</b><br>{snippet}...
                        </div>
                        """

                html += "</div>"
                st.markdown(html, unsafe_allow_html=True)

        if export_btn:
            with open("chat_export.txt", "w", encoding="utf-8") as f:
                for m in st.session_state["chat_history"]:
                    f.write(f"{m['role'].upper()}: {m['text']}\n\n")
            st.success("Exported!")

st.caption("Enterprise RAG — free edition.")
