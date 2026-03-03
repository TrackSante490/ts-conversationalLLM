import os
import shutil
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from hf_loader import load_hf_kb

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

HF_PERSIST_DIR = (PROJECT_ROOT / "vectordb" / "tracksante_hf_exam_chroma").resolve()
HF_COLLECTION_NAME = "tracksante_hf_exam"

def build_hf_vectordb():
    print("Using HF persist directory:", HF_PERSIST_DIR)
    print("Using HF collection name:", HF_COLLECTION_NAME)

    raw_docs = load_hf_kb()
    if not raw_docs:
        raise RuntimeError("No HF documents loaded.")

    # delete HF experiment DB to prevent overwriting
    if HF_PERSIST_DIR.exists():
        shutil.rmtree(HF_PERSIST_DIR)
        print(f"Deleted existing HF vectordb at {HF_PERSIST_DIR}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = splitter.split_documents(raw_docs)
    chunks = [c for c in chunks if c.page_content and c.page_content.strip()]
    print(f"Loaded {len(raw_docs)} HF docs, split into {len(chunks)} chunks")

    embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)

    vectordb = Chroma(
        persist_directory=str(HF_PERSIST_DIR),
        embedding_function=embeddings,
        collection_name=HF_COLLECTION_NAME,
    )

    vectordb.add_documents(chunks)

    print("Indexed HF dataset.")
    print("Persist dir:", HF_PERSIST_DIR)
    print("Collection:", HF_COLLECTION_NAME)

    try:
        n = vectordb._collection.count()
        print("Chroma count:", n)
    except Exception as e:
        print("Could not read count (ok):", e)

if __name__ == "__main__":
    build_hf_vectordb()