import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
INDEX_DIR = os.path.join(UPLOAD_DIR, "faiss_index")

os.makedirs(UPLOAD_DIR, exist_ok=True)

_embeddings = None
_vector_store = None

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
)


def get_embeddings():
    global _embeddings

    if _embeddings is None:
        _embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001"
        )

    return _embeddings


def _load_existing_index():
    global _vector_store

    if _vector_store is None and os.path.exists(INDEX_DIR):
        _vector_store = FAISS.load_local(
            INDEX_DIR,
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )

    return _vector_store


def process_and_index_pdf(file_path: str, filename: str):

    global _vector_store

    loader = PyPDFLoader(file_path)
    pages = loader.load()

    for page in pages:
        page.metadata["source_file"] = filename

    chunks = _splitter.split_documents(pages)

    if not chunks:
        raise ValueError(
            "No extractable text found in the PDF."
        )

    _load_existing_index()

    if _vector_store is None:
        _vector_store = FAISS.from_documents(
            chunks,
            get_embeddings()
        )
    else:
        _vector_store.add_documents(chunks)

    _vector_store.save_local(INDEX_DIR)

    return len(chunks)


def retrieve_context(question: str, k: int = 8):

    _load_existing_index()

    if _vector_store is None:
        return "", []

    docs = _vector_store.similarity_search(
        question,
        k=k
    )

    context = "\n\n".join(
        d.page_content for d in docs
    )

    sources = []

    for d in docs:
        sources.append({
            "source_file": d.metadata.get(
                "source_file",
                "unknown"
            ),
            "page": d.metadata.get(
                "page",
                "unknown"
            ),
            "text": d.page_content[:200],
        })

    return context, sources


def index_exists():

    return (
        os.path.exists(INDEX_DIR)
        or _vector_store is not None
    )
