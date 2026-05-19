"""Entry point for ``python -m RagApplication`` (Gradio web UI)."""

from RagApplication.Backend.ui import run_gradio_app

if __name__ == "__main__":
    run_gradio_app()
