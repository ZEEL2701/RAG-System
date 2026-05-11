import uuid
import logging
import os
import gradio as gr
import re
from .app import SessionBasedRAG, get_past_files

logger = logging.getLogger("enhanced_rag")


def gradio_upload_path(file):
    """Gradio 5+ passes FileData with .path; older versions may pass a path string or object with .name."""
    if file is None:
        return None
    if isinstance(file, str) and os.path.isfile(file):
        return file
    if isinstance(file, dict):
        p = file.get("path")
        return p if isinstance(p, str) and os.path.isfile(p) else None
    p = getattr(file, "path", None) or getattr(file, "name", None)
    if isinstance(p, str) and os.path.isfile(p):
        return p
    return None


def run_gradio_app():
    rag_instances = {}

    def create_rag_instance():
        session_id = str(uuid.uuid4())
        rag = SessionBasedRAG()
        if not rag.initialize():
            logger.error("Failed to initialize RAG system.")
            return None
        rag_instances[session_id] = rag
        return rag

    def get_dropdown_files(session_id):
        if not session_id or session_id not in rag_instances:
            rag = create_rag_instance()
            if not rag:
                return gr.update(choices=[], value=None), None
            session_id = list(rag_instances.keys())[-1]
        files = get_past_files()  # Show all files regardless of session
        return gr.update(choices=files, value=None), session_id

    def delete_selected_file(selected_file_dropdown, session_id=None):
        if not selected_file_dropdown:
            return "No file selected."
        match = re.search(r"ID: (\d+)\)", selected_file_dropdown)
        if not match:
            return "Invalid file selection format."
        file_id = int(match.group(1))
        rag = rag_instances.get(session_id)
        if not rag:
            return "Session not found."
        success = rag.file_manager.delete_file(file_id)
        return "File deleted successfully." if success else "Failed to delete file."

    def download_selected_file(selected_file_dropdown, session_id=None):
        if not selected_file_dropdown:
            return "No file selected."
        match = re.search(r"ID: (\d+)\)", selected_file_dropdown)
        if not match:
            return "Invalid file selection format."
        file_id = int(match.group(1))
        rag = rag_instances.get(session_id)
        if not rag:
            return "Session not found."
        file_info = rag.file_manager.get_file(file_id)
        if not file_info:
            return "File not found."
        return file_info.get("download_url", "No download URL available.")

    def upload_and_query(file, user_query, model_name, search_type, selected_file_dropdown, session_id=None):
        if not session_id or session_id not in rag_instances:
            rag = create_rag_instance()
            if not rag:
                updated_choices = get_past_files()
                return "Failed to init system.", "", [], gr.update(choices=updated_choices, value=None), session_id
            session_id = list(rag_instances.keys())[-1]
        else:
            rag = rag_instances[session_id]

        if search_type == "LLM Only":
            if not user_query.strip():
                updated_choices = get_past_files()
                return "Please enter a question.", "", [], gr.update(choices=updated_choices, value=None), session_id
            answer = rag.llm_manager.generate_response(user_query, [], model_name=model_name)
            updated_choices = get_past_files()
            return "", answer, [], gr.update(choices=updated_choices, value=None), session_id

        upload_path = gradio_upload_path(file)
        if upload_path:
            success = rag.index_document(upload_path)
            if not success:
                updated_choices = get_past_files()
                return "Failed to index document.", "", [], gr.update(choices=updated_choices, value=None), session_id
        elif selected_file_dropdown:
            try:
                file_id = int(selected_file_dropdown.split("ID: ")[1].rstrip(")"))
            except Exception:
                updated_choices = get_past_files()
                return "Invalid file selection.", "", [], gr.update(choices=updated_choices, value=None), session_id
            file_info = rag.file_manager.get_file(file_id)
            if not file_info:
                updated_choices = get_past_files()
                return "Selected file not found.", "", [], gr.update(choices=updated_choices, value=None), session_id
            file_path = file_info["file_path"]
            try:
                documents = rag.doc_processor.load_file(file_path, rag.config)
                for doc in documents:
                    doc.metadata["file_id"] = file_info["file_id"]
                    doc.metadata["source"] = file_info.get("download_url", file_path)
                rag.vector_store.add_documents(documents)
                success = True
            except Exception as e:
                logger.error(f"Error loading previously indexed doc: {e}")
                success = False
            if not success:
                updated_choices = get_past_files()
                return "Failed to load selected file.", "", [], gr.update(choices=updated_choices, value=None), session_id

        if not user_query.strip():
            updated_choices = get_past_files()
            return "Please enter a question.", "", [], gr.update(choices=updated_choices, value=None), session_id

        if rag.vector_store.get_session_document_count() == 0:
            updated_choices = get_past_files()
            return "No content found in document.", "", [], gr.update(choices=updated_choices, value=None), session_id

        if search_type == "vector":
            relevant_docs = rag.vector_store.vector_search(user_query, k=rag.config.search_k)
        elif search_type == "semantic":
            relevant_docs = rag.vector_store.semantic_search(user_query, k=rag.config.search_k)
        elif search_type == "hybrid":
            relevant_docs = rag.vector_store.hybrid_search(user_query, k=rag.config.search_k)
        else:
            updated_choices = get_past_files()
            return "Unknown search type.", "", [], gr.update(choices=updated_choices, value=None), session_id

        if not relevant_docs:
            updated_choices = get_past_files()
            return "No relevant information found.", "", [], gr.update(choices=updated_choices, value=None), session_id

        max_ctx = rag.config.max_context_documents
        if len(relevant_docs) > max_ctx:
            relevant_docs = relevant_docs[:max_ctx]

        answer = rag.llm_manager.generate_response(user_query, relevant_docs, model_name=model_name)

        sources = [{
            "label": f"{doc.metadata.get('filename', 'Unknown')} (ID: {doc.metadata.get('file_id', '-')})",
            "file_id": doc.metadata.get("file_id", None),
            "file_type": doc.metadata.get("file_type", "file"),
            "citation": doc.page_content[:200].strip() + ("..." if len(doc.page_content) > 200 else ""),
            "url": doc.metadata.get("source", "")
        } for doc in relevant_docs]

        updated_choices = get_past_files()
        return "", answer, sources, gr.update(choices=updated_choices, value=None), session_id

    with gr.Blocks(css=".gr-block { font-family: 'Segoe UI', sans-serif; padding: 8px; }") as demo:
        session_id = gr.State(None)

        gr.Markdown("### Document QA Application")

        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(label="Upload Document", file_types=["file"], scale=1)
                file_dropdown = gr.Dropdown(label="Previously Uploaded File", choices=[], interactive=True, allow_custom_value=True, scale=1)
                model_selector = gr.Dropdown(
                    choices=["llama-3.1-8b-instant", "openai/gpt-oss-120b", "meta-llama/llama-4-scout-17b-16e-instruct"],
                    value="llama-3.1-8b-instant",
                    label="Language Model"
                )
                search_selector = gr.Dropdown(
                    choices=["vector", "semantic", "hybrid", "LLM Only"],
                    value="vector",
                    label="Retrieval Strategy"
                )
                get_download_button = gr.Button("Download Selected File")
                delete_button = gr.Button("Delete Selected File")
                download_link_output = gr.Textbox(label="Download Link", interactive=False)
                delete_output = gr.Textbox(label="Delete Status", interactive=False)

            with gr.Column(scale=2):
                answer_output = gr.Textbox(label="Answer", lines=15, interactive=False)
                question_input = gr.Textbox(label="Question", placeholder="Ask a question...")
                process_button = gr.Button("Submit")

            with gr.Column(scale=1):
                sources_output = gr.JSON(label="Sources with Context")

        demo.load(fn=get_dropdown_files, inputs=[session_id], outputs=[file_dropdown, session_id])
        process_button.click(upload_and_query,
                             inputs=[file_input, question_input, model_selector, search_selector, file_dropdown, session_id],
                             outputs=[answer_output, answer_output, sources_output, file_dropdown, session_id])
        get_download_button.click(download_selected_file, inputs=[file_dropdown, session_id], outputs=[download_link_output])
        delete_button.click(delete_selected_file, inputs=[file_dropdown, session_id], outputs=[delete_output])

        share_public = os.getenv("GRADIO_SHARE", "true").lower() not in ("0", "false", "no")
        logger.info(
            f"Launching Gradio on 127.0.0.1:7860 (share={share_public}; public link prints below if enabled)"
        )
        demo.launch(server_name="127.0.0.1", server_port=7860, share=share_public)

if __name__ == "__main__":
    import sys
    try:
        run_gradio_app()
    except Exception as e:
        print(f"Exception occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
