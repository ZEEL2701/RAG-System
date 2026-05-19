# RAG application

Modular backend under `Backend/`. Configure PostgreSQL, Groq, and optional S3 via environment variables (see `.env.example` if present, or create `.env` using `python -m RagApplication.Backend.cli setup` from the folder where you want `.env` created).

## Prerequisites

- Python 3.10+
- PostgreSQL with pgvector
- Groq API key

## Install

From the **repository root** (the directory that contains the `RagApplication` folder):

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r RagApplication/requirements.txt
```

Copy or create `.env` next to your working directory (the CLI `setup` command writes `.env` in the **current working directory**). For production, keep secrets in `.env` and **never** commit them.

## Run the Gradio UI (primary entry point)

From the repository root:

```bash
python -m RagApplication
```

Equivalent:

```bash
python -m RagApplication.Backend.ui
```

## CLI (index, query, files, interactive)

```bash
python -m RagApplication.Backend.cli --help
python -m RagApplication.Backend.cli serve
python -m RagApplication.Backend.cli index path\to\file.pdf
python -m RagApplication.Backend.cli query "Your question"
```

`serve` starts the same Gradio app as `python -m RagApplication`.

## Frontend

The Vite/React app lives under `Frontend/` (see `Frontend/Rag_Frontend/README.md`).

## Legacy

`legacy/final.py` is a preserved alternate monolith for reference only; the supported stack is `Backend/`.
