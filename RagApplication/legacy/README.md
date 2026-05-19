# Legacy monolith (reference only)

`final.py` was moved here from the repository root. It is an **alternate** single-file implementation (chat-style Gradio UI, different dropdown labels, optional FAISS/BM25 helpers in that file).

The supported application is **`RagApplication/Backend/`** (modular `app.py`, `ui.py`, `vector_store.py`, etc.). Do not import this legacy file from the modular app.

To inspect old behavior, open `final.py` in this folder; it is not wired to any launcher.
