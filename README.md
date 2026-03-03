# TrackSanté — Conversational RAG LLM

This repository contains a **conversational, retrieval-augmented LLM agent** designed around the TrackSanté concept of guided, health-oriented coaching. The system combines:

- a **local knowledge base** 
- a **vector retrieval layer** (to fetch relevant passages + sources)
- a **conversational agent** (that maintains chat history + user context across sessions)
- an LLM backend (currently configured for **Ollama via LangChain**)

---

## What’s in this repo

- **`rag_agent.py`** — The main agent. A CLI-based conversational RAG assistant that:
  - loads persisted session state (chat history + user context)
  - retrieves relevant knowledge snippets on each user turn
  - generates an answer
  - updates and persists state for continuity across runs

- **`kb_loader.py`** — Knowledge base loading utilities (parsing/assembling documents for indexing).

- **`kb_indexer.py`** — Knowledge base indexing utilities (building/updating the vector store).

- **`knowledge_base/`** — The knowledge content used for retrieval.

- **`requirements.txt`** — Python dependencies.

---

## Key capabilities

### Conversational, multi-turn interaction
The agent runs as an interactive CLI and tracks multi-turn conversation state.

### Retrieval-Augmented Generation (RAG)
On each user message, the agent retrieves relevant knowledge base context and uses it to ground the response.

### Session persistence
The agent persists:
- **chat history**
- **user context** (collected facts)

---

## Setup

### 1) Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows PowerShell

pip install -r requirements.txt
```

### 2) Configure the LLM backend (Ollama)

Ensure Ollama is running (local or remote).

Environment variables used by the scripts:

- `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `OLLAMA_CHAT_MODEL` (default: `llama3.1`)
- `OLLAMA_EMBED_MODEL` (default: `nomic-embed-text`)

Example:

```bash
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_CHAT_MODEL="llama3.1"
export OLLAMA_EMBED_MODEL="nomic-embed-text"
```

If needed, pull the models:

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

---

## Running the agent

```bash
python rag_agent.py
```

You can exit at any time with:

- `exit`
- `quit`

---

## Knowledge base & indexing workflow

The knowledge base content lives in:

- `knowledge_base/`

Typical iteration loop:
1. Update/add domain documents in `knowledge_base/`
2. Rebuild or update the vector index using `kb_loader.py` and `kb_indexer.py`
3. Run `rag_agent.py`

---

## Notes / troubleshooting

- **Can’t connect to Ollama**: verify `OLLAMA_BASE_URL` and that `ollama serve` is running.
- **Model not found**: run `ollama pull <model>`.
- **Retrieval seems irrelevant**: confirm the KB has been indexed and embeddings are being created correctly.

---
