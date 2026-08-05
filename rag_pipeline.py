"""
Handles PDF loading, chunking, embedding generation, and the FAISS vector store.
Uses ONE SHARED INDEX - every uploaded PDF is added to the same knowledge base.
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
INDEX_DIR = os.path.join(UPLOAD_DIR, "faiss_index")

os.makedirs(UPLOAD_DIR, exist_ok=True)

_embeddings = None  # loaded lazily, on first actual use (not at import time)

_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)

_vector_store = None  # loaded lazily / built on first upload


def get_embeddings():
    """Lazily create the embeddings model. Avoids downloading/loading it at import time,
    which can block app startup on Render."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _embeddings


def _load_existing_index():
    global _vector_store
    if _vector_store is None and os.path.exists(INDEX_DIR):
        _vector_store = FAISS.load_local(
            INDEX_DIR, get_embeddings(), allow_dangerous_deserialization=True
        )
    return _vector_store


def process_and_index_pdf(file_path: str, filename: str):
    """Load a PDF, chunk it, embed it, and add it to the shared FAISS index."""
    global _vector_store

    loader = PyPDFLoader(file_path)
    pages = loader.load()

    # tag each page with the source filename for citations later
    for p in pages:
        p.metadata["source_file"] = filename

    chunks = _splitter.split_documents(pages)

    if not chunks:
        raise ValueError(
            f"No extractable text found in '{filename}'. "
            "This usually means the PDF is scanned/image-only, or has no selectable text."
        )

    _load_existing_index()

    if _vector_store is None:
        _vector_store = FAISS.from_documents(chunks, get_embeddings())
    else:
        _vector_store.add_documents(chunks)

    _vector_store.save_local(INDEX_DIR)
    return len(chunks)


def retrieve_context(question: str, k: int = 8):
    """Return top-k relevant chunks for a question, plus source info."""
    _load_existing_index()

    if _vector_store is None:
        return "", []

    results = _vector_store.similarity_search(question, k=k)

    context_text = "\n\n".join(doc.page_content for doc in results)
    sources = [
        {
            "source_file": doc.metadata.get("source_file", "unknown"),
            "page": doc.metadata.get("page", "unknown"),
            "text": doc.page_content[:200],
        }
        for doc in results
    ]
    return context_text, sources


def index_exists() -> bool:
    return os.path.exists(INDEX_DIR) or _vector_store is not None
