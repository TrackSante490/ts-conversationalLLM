import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from kb_loader import load_markdown_kb
from pathlib import Path
import shutil

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
PERSIST_DIR = (PROJECT_ROOT / "vectordb" / "tracksante_chroma").resolve()

def build_vectordb():
    raw_docs = load_markdown_kb()
    if not raw_docs:
        raise RuntimeError("No knowledge base documents loaded.")
    
    if PERSIST_DIR.exists():
        shutil.rmtree(PERSIST_DIR)
        print(f"Deleted existing vectordb at {PERSIST_DIR}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200
    )
    docs = splitter.split_documents(raw_docs)
    if not docs:
        raise RuntimeError("No non-empty chunks created. Check your markdown files.")

    docs = [d for d in docs if d.page_content and d.page_content.strip()]
    print(f"Loaded {len(raw_docs)} docs, split into {len(docs)} non-empty chunks")
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    
    vectordb = Chroma(
        persist_directory=str(PERSIST_DIR),
        embedding_function=embeddings,
        collection_name="tracksante_policy"
    )

    vectordb.add_documents(docs)

    print(f"Indexed {len(docs)} chunks")

if __name__ == "__main__":
    build_vectordb()