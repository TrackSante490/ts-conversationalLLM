import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from pathlib import Path

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
PERSIST_DIR = (PROJECT_ROOT / "vectordb" / "tracksante_chroma").resolve()

def get_retriever():
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)

    vectordb = Chroma(
        collection_name="tracksante_kb",
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIR)
    )

    return vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 12}
    )
