import chromadb
from sentence_transformers import SentenceTransformer


class VectorRetriever:
    def __init__(self, persist_directory="data/chroma_store"):

        # NEW CLIENT (NO SETTINGS REQUIRED)
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Create or load collection
        self.collection = self.client.get_or_create_collection(
            name="rag_collection",
            metadata={"hnsw:space": "cosine"}
        )

        # Embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def add_documents(self, docs):
        embeddings = self.model.encode(docs, convert_to_numpy=True)
        ids = [f"chunk_{i}" for i in range(len(docs))]

        self.collection.add(
            ids=ids,
            documents=docs,
            embeddings=embeddings.tolist()
        )

    def search(self, query, k=5):
        emb = self.model.encode([query], convert_to_numpy=True)

        result = self.collection.query(
            query_embeddings=emb.tolist(),
            n_results=k
        )

        docs = result.get("documents", [[]])[0]
        return docs
