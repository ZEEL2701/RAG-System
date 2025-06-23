import os
import logging
import uuid
from typing import List, Dict, Any, Optional

from .config import Config
from .file_manager import FileManager
from .s3_manager import S3Manager
from .document_processor import DocumentProcessor
from .vector_store import SessionVectorStore
from .llm_manager import LLMManager
from langchain_core.documents import Document
from .web_search import fetch_web_snippets

logger = logging.getLogger("enhanced_rag")

def get_past_files(session_id: Optional[str] = None) -> List[str]:
    """Helper to fetch past files for dropdowns etc."""
    from psycopg2.extras import RealDictCursor
    import psycopg2

    cfg = Config()
    try:
        conn = psycopg2.connect(cfg.connection_string)
    except Exception as e:
        logger.error(f"[Startup] Failed to connect to database: {e}")
        return []

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if session_id:
                cur.execute("""
                    SELECT id, file_name, file_path, upload_date
                    FROM file_registry
                    WHERE session_id = %s
                    ORDER BY id DESC;
                """, (session_id,))
            else:
                cur.execute("""
                    SELECT id, file_name, file_path, upload_date
                    FROM file_registry
                    ORDER BY id DESC;
                """)
            rows = cur.fetchall()
            logger.info(f"[Startup] Found {len(rows)} file entries in the database.")

            choices = []
            for r in rows:
                if os.path.exists(r["file_path"]):
                    choices.append(f"{r['file_name']} (ID: {r['id']})")
                else:
                    logger.warning(f"Skipping missing file: {r['file_path']}")

            logger.info(f"[Startup] Returning {len(choices)} valid files for dropdown.")
            return choices
    except Exception as e:
        logger.error(f"[Startup] Error fetching files: {e}")
        return []
    finally:
        conn.close()

class SessionBasedRAG:
    def __init__(self):
        self.config = Config()
        self.s3_manager = S3Manager(self.config)
        self.file_manager = FileManager(self.config, self.s3_manager)
        self.doc_processor = DocumentProcessor()
        self.vector_store = SessionVectorStore(self.config)
        self.llm_manager = LLMManager(self.config)

    def initialize(self) -> bool:
        try:
            if not self.config.validate():
                logger.error("Configuration validation failed.")
                return False
            self.vector_store.initialize()
            self.llm_manager.initialize()
            logger.info("RAG system initialized successfully.")
            return True
        except Exception as e:
            logger.error(f"RAG initialization error: {e}")
            return False

    def index_document(self, file_path: str) -> bool:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False
        try:
            file_info = self.file_manager.store_file(file_path, self.vector_store.session_id)
            if not file_info:
                logger.error(f"Failed to store file: {file_path}")
                return False

            logger.info(f"Indexing document: {file_path}")
            documents = self.doc_processor.load_file(file_path, self.config)
            for doc in documents:
                doc.metadata["file_id"] = file_info["file_id"]
                doc.metadata["source"] = file_info.get("download_url", file_path)
            self.vector_store.add_documents(documents)
            logger.info(f"Document indexed successfully: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Indexing error: {e}")
            return False

    def list_files(self, session_only=False, session_id=None):
        return self.file_manager.list_files(session_id=session_id if session_only else None)

    def download_file(self, file_id: int, destination_path: str) -> bool:
        file_info = self.file_manager.get_file(file_id)
        if not file_info:
            return False
        if os.path.exists(file_info["file_path"]):
            try:
                import shutil
                shutil.copy2(file_info["file_path"], destination_path)
                return True
            except Exception as e:
                logger.error(f"Failed to copy file: {e}")
        if self.config.s3_enabled and "object_key" in file_info:
            return self.s3_manager.download_file(file_info["object_key"], destination_path)
        return False

    def query(self, question: str, model_name: Optional[str] = None, strategy: str = "vector" , include_web: bool = False) -> Dict[str, Any]:
        try:
            logger.info(f"Query using strategy: {strategy}")

            if strategy == "LLM Only":
                logger.info("Using LLM Only mode, no document context.")
                answer = self.llm_manager.generate_response(question, [], model_name=model_name)
                return {"preview": [], "answer": answer, "sources": []}

            docs_with_scores = self.vector_store.similarity_search_with_score(question, k=self.config.search_k)

            if not docs_with_scores:
                logger.warning("No documents retrieved from vector store.")
                return {"preview": [], "answer": "No relevant information found.", "sources": []}

            if strategy == "vector":
                relevant_docs = [doc for doc, _ in docs_with_scores[:self.config.max_context_documents]]
            elif strategy == "semantic":
                relevant_docs = self.llm_manager.rank_relevance(question, docs_with_scores[:10])
            elif strategy == "hybrid":
                vector_top = [doc for doc, _ in docs_with_scores[:3]]
                reranked_top = self.llm_manager.rank_relevance(question, docs_with_scores[:10])
                combined = vector_top + reranked_top
                seen = set()
                relevant_docs = []
                for doc in combined:
                    uid = (doc.metadata.get("filename", ""), doc.metadata.get("chunk", ""))
                    if uid not in seen:
                        seen.add(uid)
                        relevant_docs.append(doc)
                relevant_docs = relevant_docs[:self.config.max_context_documents]
            else:
                return {"preview": [], "answer": f"Unknown search type '{strategy}'", "sources": []}
            if include_web:
                web_docs = fetch_web_snippets(question, k=3)
                relevant_docs.extend(web_docs)

            preview = []
            seen_sources = set()
            for doc in relevant_docs:
                source = doc.metadata.get("source", "")
                if source and source not in seen_sources:
                    preview.append({
                        "filename": doc.metadata.get("filename", "Unknown"),
                        "file_type": doc.metadata.get("file_type", "document"),
                        "url": source,
                        "excerpt": doc.page_content[:1000] + "..."
                    })
                    seen_sources.add(source)

            answer = self.llm_manager.generate_response(question, relevant_docs, model_name=model_name)
            sources = [{"url": p["url"], "filename": p["filename"], "file_type": p["file_type"]} for p in preview]
            summary = f"_Answer based on {len(preview)} sources{' including web results' if include_web else ''}._"

            return {
                "preview": preview,
                "answer": f"{answer}\n\n{summary}",
                "sources": sources
            }

        except Exception as e:
            logger.error(f"Query error: {e}")
            return {"preview": [], "answer": "An error occurred during the query.", "sources": []}
