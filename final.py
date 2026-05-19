import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import argparse
from pathlib import Path
import pandas as pd
import gradio as gr
import boto3
from botocore.exceptions import ClientError
import uuid
import requests  # Added for connection testing

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    PyMuPDFLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import PGVector
from sentence_transformers import SentenceTransformer
from groq import Groq

import psycopg2
from psycopg2.extras import RealDictCursor

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever


rag_instances = {}

def get_past_files() -> List[str]:
    cfg = Config()
    conn = psycopg2.connect(cfg.connection_string)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, file_name, upload_date
                FROM file_registry
                ORDER BY upload_date DESC;
            """)
            rows = cur.fetchall()
    finally:
        conn.close()

    choices = []
    for r in rows:
        ts = r["upload_date"].strftime("%Y-%m-%d %H:%M")
        full_name = r["file_name"]
        # Strip session prefix: split on first underscore if present
        name_only = full_name.split("_", 1)[1] if "_" in full_name else full_name
        choices.append(f"{name_only} | Uploaded: {ts} | ID: {r['id']}")
    return choices


def smart_upload_or_query(file, selected_past_file, question, model_name, retrieval_strategy, session_id, chat_history):
    if session_id is None:
        session_id = str(uuid.uuid4())

    if session_id not in rag_instances:
        rag_instances[session_id] = SessionBasedRAG()
        if not rag_instances[session_id].initialize():
            raise RuntimeError("Failed to initialize RAG system")

    rag = rag_instances[session_id]

    if retrieval_strategy == "LLM Only":
        result = rag.query(question, model_name=model_name, strategy="LLM Only")
        answer = "".join(chunk or "" for chunk in result["answer"])
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": answer})

        return "", chat_history, [], gr.update(choices=get_past_files(), value=None), session_id

    if file is not None:
        temp_path = file.name
        os.makedirs("uploaded_files", exist_ok=True)
        success = rag.index_document(temp_path)
        if not success:
            return "Failed to index uploaded document.", chat_history, [], None, session_id
        result = rag.query(question, model_name=model_name, strategy=retrieval_strategy)

    elif selected_past_file:
        file_id = int(selected_past_file.split("ID:")[1].strip())
        result = rag.retrieve_by_file_id(file_id, question, model_name=model_name, strategy=retrieval_strategy)
    else:
        return "Please upload a document or select one from past documents.", chat_history, [], None, session_id

    preview_text = "\n\n".join([f"{p['filename']}:\n{p['excerpt']}" for p in result["preview"]])
    answer = "".join(result["answer"])
    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": answer})


    updated_choices = get_past_files()
    selected_value = selected_past_file if selected_past_file else (
    updated_choices[0] if updated_choices else None
    )
    return preview_text, chat_history, result["sources"], gr.update(choices=updated_choices, value=selected_value), session_id




load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("enhanced_rag")

# Configuration class
class Config:
    """Configuration for the RAG system."""

    def __init__(self):
        self.db_host = os.getenv("DB_HOST", "localhost")
        self.db_port = os.getenv("DB_PORT", "5432")
        self.db_name = os.getenv("DB_NAME", "ragdb")
        self.db_user = os.getenv("DB_USER", "postgres")
        self.db_password = os.getenv("DB_PASSWORD", "Zeel2701")

        # Ollama configuration
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

        # Document processing
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))
        self.collection_name = os.getenv("COLLECTION_NAME", "rag-pgvector")

        self.max_tokens = int(os.getenv("MAX_TOKENS", "2048"))

        # Groq API configuration
        self.groq_base_url = "https://api.groq.com"
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

        # RAG configuration
        self.max_context_documents = int(os.getenv("MAX_CONTEXT_DOCUMENTS", "7"))
        self.search_k = int(os.getenv("SEARCH_K", "10"))  # Retrieve more and filter

        # AWS S3 configuration
        self.s3_enabled = os.getenv("S3_ENABLED", "False").lower() == "true"
        self.s3_bucket_name = os.getenv("S3_BUCKET_NAME", "")
        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID", "")
        self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", "")
        self.aws_region = os.getenv("AWS_REGION", "us-east-1")

    def setup_file_registry_db(self):
        """Set up the file registry database."""
        import psycopg2

        conn = psycopg2.connect(self.connection_string)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS file_registry (
                    id SERIAL PRIMARY KEY,
                    file_name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    object_key TEXT,
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    session_id TEXT,
                    metadata JSONB DEFAULT '{}'::jsonb
                );
            """)
            conn.commit()
            logger.info("File registry table created or already exists")
        except Exception as e:
            logger.error(f"Failed to create file registry table: {str(e)}")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()

    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    def validate(self) -> bool:
        if not all([self.db_host, self.db_name, self.db_user]):
            logger.error("Database configuration incomplete")
            return False
        if not self.ollama_base_url:
            logger.error("Ollama base URL not configured")
            return False
        if not self.groq_api_key:
            logger.error("Groq API key is missing")
            return False
        if self.s3_enabled:
            if not all([self.s3_bucket_name, self.aws_access_key_id, self.aws_secret_access_key]):
                logger.error("S3 configuration is incomplete.")
                return False
        return True

# New FileManager class
class FileManager:
    def __init__(self, config: Config, s3_manager: 'S3Manager'):
        self.config = config
        self.s3_manager = s3_manager
        self.local_storage_path = os.path.join(os.getcwd(), "uploaded_files")
        os.makedirs(self.local_storage_path, exist_ok=True)
        self.config.setup_file_registry_db()  # Corrected line

    def register_file(self, file_path: str, session_id: str, object_key: str = None, metadata: Dict = None):
        """Register a file in the database."""
        import psycopg2
        import json

        file_name = os.path.basename(file_path)
        file_type = os.path.splitext(file_name)[1].lower()[1:]  # Remove the dot

        conn = psycopg2.connect(self.config.connection_string)
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                INSERT INTO file_registry
                (file_name, file_path, file_type, object_key, session_id, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id;
                """,
                (
                    file_name,
                    file_path,
                    file_type,
                    object_key,
                    session_id,
                    json.dumps(metadata or {})
                )
            )
            file_id = cursor.fetchone()[0]
            conn.commit()
            logger.info(f"Registered file with ID {file_id}: {file_name}")
            return file_id
        except Exception as e:
            logger.error(f"Failed to register file: {str(e)}")
            conn.rollback()
            return None
        finally:
            cursor.close()
            conn.close()

    def store_file(self, source_path: str, session_id: str, metadata: Dict = None) -> Dict:
        """Store a file and register it."""
        # Make a copy in local storage
        file_name = os.path.basename(source_path)
        local_path = os.path.join(self.local_storage_path, f"{session_id}_{file_name}")

        try:
            import shutil
            shutil.copy2(source_path, local_path)
            logger.info(f"Stored local copy at {local_path}")

            # Upload to S3 if enabled
            object_key = None
            if self.config.s3_enabled:
                object_key = self.s3_manager.upload_file(source_path)

            # Register in database
            file_id = self.register_file(
                local_path,
                session_id,
                object_key,
                metadata
            )

            result = {
                "file_id": file_id,
                "file_name": file_name,
                "local_path": local_path,
                "object_key": object_key,
                "session_id": session_id
            }

            if object_key:
                result["download_url"] = self.s3_manager.generate_presigned_url(object_key)

            return result
        except Exception as e:
            logger.error(f"Failed to store file: {str(e)}")
            return None

    def list_files(self, session_id: str = None) -> List[Dict]:
        """List files from the registry, optionally filtered by session_id."""
        import psycopg2

        conn = psycopg2.connect(self.config.connection_string)
        cursor = conn.cursor()

        try:
            if session_id:
                cursor.execute(
                    """
                    SELECT id, file_name, file_path, file_type, object_key, upload_date, metadata
                    FROM file_registry
                    WHERE session_id = %s
                    ORDER BY upload_date DESC;
                    """,
                    (session_id,)
                )
            else:
                cursor.execute(
                    """
                    SELECT id, file_name, file_path, file_type, object_key, upload_date, metadata
                    FROM file_registry
                    ORDER BY upload_date DESC;
                    """
                )

            files = []
            for row in cursor.fetchall():
                file_id, file_name, file_path, file_type, object_key, upload_date, metadata = row
                file_info = {
                    "file_id": file_id,
                    "file_name": file_name,
                    "file_path": file_path,
                    "file_type": file_type,
                    "upload_date": upload_date.strftime("%Y-%m-%d %H:%M:%S")
                }

                if object_key:
                    file_info["object_key"] = object_key
                    try:
                        file_info["download_url"] = self.s3_manager.generate_presigned_url(object_key)
                    except Exception as e:
                        logger.error(f"Failed to generate presigned URL for {file_name}: {str(e)}")

                files.append(file_info)

            return files
        except Exception as e:
            logger.error(f"Failed to list files: {str(e)}")
            return []
        finally:
            cursor.close()
            conn.close()

    def get_file(self, file_id: int) -> Dict:
        """Get details for a specific file."""
        import psycopg2

        conn = psycopg2.connect(self.config.connection_string)
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                SELECT id, file_name, file_path, file_type, object_key, upload_date, metadata
                FROM file_registry
                WHERE id = %s;
                """,
                (file_id,)
            )

            row = cursor.fetchone()
            if not row:
                return None

            file_id, file_name, file_path, file_type, object_key, upload_date, metadata = row
            file_info = {
                "file_id": file_id,
                "file_name": file_name,
                "file_path": file_path,
                "file_type": file_type,
                "upload_date": upload_date.strftime("%Y-%m-%d %H:%M:%S"),
                "metadata": metadata
            }

            if object_key:
                file_info["object_key"] = object_key
                try:
                    file_info["download_url"] = self.s3_manager.generate_presigned_url(object_key)
                except Exception as e:
                    logger.error(f"Failed to generate presigned URL for {file_name}: {str(e)}")

            return file_info
        except Exception as e:
            logger.error(f"Failed to get file info: {str(e)}")
            return None
        finally:
            cursor.close()
            conn.close()

    def delete_file(self, file_id: int) -> bool:
        """Delete a file from storage and registry."""
        file_info = self.get_file(file_id)
        if not file_info:
            return False

        # Delete from S3 if applicable
        if "object_key" in file_info and self.config.s3_enabled:
            self.s3_manager.delete_file(file_info["object_key"])

        # Delete local file
        if os.path.exists(file_info["file_path"]):
            try:
                os.remove(file_info["file_path"])
            except Exception as e:
                logger.error(f"Failed to delete local file: {str(e)}")

        # Remove from registry
        import psycopg2

        conn = psycopg2.connect(self.config.connection_string)
        cursor = conn.cursor()

        try:
            cursor.execute(
                "DELETE FROM file_registry WHERE id = %s;",
                (file_id,)
            )
            conn.commit()
            logger.info(f"Deleted file with ID {file_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete file from registry: {str(e)}")
            conn.rollback()
            return False
        finally:
            cursor.close()
            conn.close()

# S3 Manager
class S3Manager:
    def __init__(self, config: Config):
        self.config = config
        self.s3_client = None
        if self.config.s3_enabled:
            self.s3_client = boto3.client(
                "s3",
                region_name=self.config.aws_region,
                aws_access_key_id=self.config.aws_access_key_id,
                aws_secret_access_key=self.config.aws_secret_access_key
            )

    def upload_file(self, file_path: str) -> Optional[str]:
        if not self.s3_client:
            logger.info("S3 not enabled; skipping upload.")
            return None
        try:
            file_name = os.path.basename(file_path)
            object_key = f"documents/{file_name}"
            self.s3_client.upload_file(
                Filename=file_path,
                Bucket=self.config.s3_bucket_name,
                Key=object_key
            )
            logger.info(f"Uploaded to S3: s3://{self.config.s3_bucket_name}/{object_key}")
            return object_key
        except ClientError as e:
            logger.error(f"S3 upload error: {str(e)}")
            return None

    def generate_presigned_url(self, object_key: str, expiration: int = 3600) -> Optional[str]:
        if not self.s3_client or not object_key:
            return None
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.config.s3_bucket_name, 'Key': object_key},
                ExpiresIn=expiration
            )
            return url
        except ClientError as e:
            logger.error(f"Presigned URL error: {str(e)}")
            return None

# Document Processor
class DocumentProcessor:
    @staticmethod
    def load_file(file_path, config: Config) -> List[Document]:
        ext = os.path.splitext(file_path)[1].lower()
        try:
            filename = os.path.basename(file_path)
            if ext == ".txt":
                loader = TextLoader(file_path)
                docs = loader.load()
            elif ext == ".csv":
                df = pd.read_csv(file_path)
                text_content = DocumentProcessor._dataframe_to_text(df)
                docs = [Document(page_content=text_content, metadata={"source": file_path, "filename": filename})]
            elif ext in [".xlsx", ".xls"]:
                df = pd.read_excel(file_path)
                text_content = DocumentProcessor._dataframe_to_text(df)
                docs = [Document(page_content=text_content, metadata={"source": file_path, "filename": filename})]
            elif ext == ".pdf":
                loader = PyMuPDFLoader(file_path)
                docs = loader.load()
            elif ext in [".doc", ".docx"]:
                loader = UnstructuredWordDocumentLoader(file_path)
                docs = loader.load()
            elif ext in [".ppt", ".pptx"]:
                loader = UnstructuredPowerPointLoader(file_path)
                docs = loader.load()
            else:
                raise ValueError(f"Unsupported file type: {ext}")

            for doc in docs:
                doc.metadata["filename"] = filename
                doc.metadata["file_type"] = ext[1:]  # Remove the dot

            return DocumentProcessor.split_documents(docs, config)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            raise

    @staticmethod
    def _dataframe_to_text(df: pd.DataFrame) -> str:
        text_parts = ["Columns: " + ", ".join(df.columns)]
        for idx, row in df.iterrows():
            row_text = [f"{col}: {row[col]}" for col in df.columns]
            text_parts.append(f"Row {idx}: {' | '.join(row_text)}")
        return "\n".join(text_parts)

    @staticmethod
    def split_documents(documents: List[Document], config: Config) -> List[Document]:
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap
            )
            split_docs = []
            for doc in documents:
                chunks = splitter.split_text(doc.page_content)
                for i, chunk in enumerate(chunks):
                    chunk_metadata = doc.metadata.copy()
                    chunk_metadata["chunk"] = i + 1
                    chunk_metadata["total_chunks"] = len(chunks)
                    split_docs.append(Document(page_content=chunk, metadata=chunk_metadata))
            logger.info(f"Split {len(documents)} documents into {len(split_docs)} chunks")
            return split_docs
        except Exception as e:
            logger.error(f"Splitting error: {str(e)}")
            raise

class HuggingFaceEmbeddings:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False)

    def embed_query(self, text):
        return self.model.encode([text])[0]

# Vector Store with Ollama Connection Testing
class SessionVectorStore:

    def __init__(self, config: Config):
        self.config = config
        self.embeddings = None
        self.vectorstore = None
        self.session_id = None
        self.session_docs = []
        self.bm25_docs = []

    def initialize(self, session_id: Optional[str] = None):
        try:
            logger.info(f"Initializing embeddings using model {self.config.embedding_model}")
            self.embeddings = HuggingFaceEmbeddings(self.config.embedding_model)
            logger.info("Embeddings initialized successfully.")

            if session_id:
                self.session_id = session_id
            else:
                self.session_id = str(uuid.uuid4())

            session_collection = self.config.collection_name   # <<<<< fix: permanent
            logger.info(f"Using collection: {session_collection}")

            self.vectorstore = PGVector(
                collection_name=session_collection,
                embedding_function=self.embeddings,
                connection_string=self.config.connection_string
            )
        except Exception as e:
            logger.error(f"Vector store initialization error: {str(e)}")
            raise



    def load_documents_by_file_id(self, file_id: int, k: int = 100) -> List[Document]:
        try:
            docs = self.vectorstore.similarity_search(
                query="",  # empty query, just filter
                k=k,
                filter={"file_id": str(file_id)}
            )
            logger.info(f"Loaded {len(docs)} chunks for file_id {file_id}")
            return docs
        except Exception as e:
            logger.error(f"Error loading docs by file_id: {str(e)}")
            return []


    def add_documents(self, documents: List[Document]):
        try:
            if not documents:
                logger.warning("No documents to add.")
                return
            if not self.embeddings:
                self.initialize()

            for doc in documents:
                doc.metadata["session_id"] = self.session_id

            self.vectorstore.add_documents(documents)
            self.session_docs.extend(documents)
            self.bm25_docs.extend(documents)
            logger.info(f"Added {len(documents)} docs to session store. Total: {len(self.session_docs)}")
        except Exception as e:
            logger.error(f"Error adding docs: {str(e)}")
            raise

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        try:
            if not self.vectorstore:
                logger.warning("Vector store not initialized.")
                return []
            docs = self.vectorstore.similarity_search(query, k=k)
            logger.info(f"Found {len(docs)} docs for query: {query[:50]}...")
            return docs
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return []

    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        try:
            if not self.vectorstore:
                logger.warning("Vector store not initialized.")
                return []
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
            logger.info(f"Found {len(docs_with_scores)} docs with scores for query: {query[:50]}...")
            return docs_with_scores
        except Exception as e:
            logger.error(f"Search with score error: {str(e)}")
            return []

    def get_session_document_count(self) -> int:
        """Return the number of documents in the current session."""
        return len(self.session_docs)
    
    def bm25_search(self, query: str, k: int = 5) -> List[Document]:
        if not self.bm25_docs:
            logger.warning("BM25: No documents to search.")
            return []
        retriever = BM25Retriever.from_documents(self.bm25_docs)
        return retriever.get_relevant_documents(query)[:k]
# LLM Manager
class LLMManager:
    def __init__(self, config: Config):
        self.config = config
        self.client = None

    def initialize(self):
        try:
            if not self.config.groq_api_key:
                logger.error("Groq API key missing")
                raise ValueError("Groq API key required")
            self.client = Groq(
                api_key=self.config.groq_api_key,
                base_url=self.config.groq_base_url
            )
            print("self.config.groq_base_url: ", self.config.groq_base_url)
            logger.info(f"Groq client initialized with model: {self.config.groq_model}")
            try:
                models = self.client.models.list()
                logger.info(f"Successfully connected to Groq API. Models available.")
            except Exception as e:
                logger.error(f"Failed to list Groq models, but continuing: {str(e)}")

        except Exception as e:
            logger.error(f"LLM initialization error: {str(e)}")
            raise

    def generate_response(self, query: str, context_docs: List[Document], model_name: Optional[str] = None):
        if not self.client:
            self.initialize()

        chosen_model = model_name or self.config.groq_model

        context_sections = []
        for i, doc in enumerate(context_docs):
            metadata = doc.metadata
            source = metadata.get("filename", "Unknown")
            file_type = metadata.get("file_type", "document")
            chunk = metadata.get("chunk", "")
            total_chunks = metadata.get("total_chunks", "")
            chunk_info = f" (Chunk {chunk}/{total_chunks})" if chunk and total_chunks else ""

            section = f"[DOCUMENT {i+1}]\nSource: {source}{chunk_info}\nType: {file_type}\nContent:\n{doc.page_content}\n"
            context_sections.append(section)

        context = "\n".join(context_sections)

        system_prompt = (
            "You are a helpful  assistant that provides accurate and well formated information based on the context provided. "
            "Follow these rules when generating answers:\n"
            "1. Mostly use information from the provided context documents.\n"
            "2. If the context doesn't contain information needed to fully answer the question, say so clearly.\n"
            "3. Do not invent or assume information that's not in the context.\n"
            "4. Cite the specific document sources when providing information.\n"
            "5. Format your answer for clarity and readability.\n"
            "6. Format the answer professionally using structured bullet points when necessary.\n"
        )

        user_prompt = (
            f"Question: {query}\n\n"
            f"Context Documents:\n{context}\n\n"
            f"Please provide a comprehensive answer based solely on the provided context documents. "
            f"Cite specific documents when providing information."
            f"Format your response for clarity using sections, or short paragraphs."
        )

        
        try:
            completion = self.client.chat.completions.create(
                model=chosen_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=2048,
                stream=True,
            )

            for chunk in completion:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

        except Exception as e:
            logger.error(f"Streaming response error: {str(e)}")
            yield "An error occurred during answer generation."


    def rank_relevance(self, query: str, docs_with_scores: List[tuple]) -> List[Document]:
        """Rank documents by relevance and filter out irrelevant ones."""
        try:
            if not self.client:
                self.initialize()
            if not docs_with_scores:
                return []

            threshold_score = 0.3  
            filtered_docs = [(doc, score) for doc, score in docs_with_scores if score < threshold_score]

            if not filtered_docs:
                filtered_docs = sorted(docs_with_scores, key=lambda x: x[1])[:3]

            if len(filtered_docs) > self.config.max_context_documents:
                ranked_docs = []
                for doc, score in filtered_docs:
                    system_prompt = (
                        "You are a document relevance ranker. Your task is to score how relevant a document is to a query.\n"
                        "Rate the relevance on a scale of 0-10, where 10 is perfectly relevant and 0 is completely irrelevant.\n"
                        "Your response must be exactly one number between 0 and 10."
                    )
                    user_prompt = (
                        f"Query: {query}\n\n"
                        f"Document content:\n{doc.page_content}\n\n"
                        f"Rate the relevance of this document to the query on a scale of 0-10:"
                    )

                    response = self.client.chat.completions.create(
                        model=self.config.groq_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.1,
                        max_tokens=self.config.max_tokens,
                    )

                    try:
                        relevance_score = float(response.choices[0].message.content.strip())
                    except ValueError:
                        relevance_score = 5.0

                    ranked_docs.append((doc, relevance_score))

                sorted_docs = sorted(ranked_docs, key=lambda x: x[1], reverse=True)
                result = [doc for doc, _ in sorted_docs[:self.config.max_context_documents]]
                logger.info(f"LLM ranked {len(filtered_docs)} docs; using top {len(result)} relevant docs.")
                return result
            else:
                return [doc for doc, _ in filtered_docs]
        except Exception as e:
            logger.error(f"Ranking error: {str(e)}")
            return [doc for doc, _ in docs_with_scores[:self.config.max_context_documents]]

# RAG Application
class SessionBasedRAG:
    def __init__(self):
        self.config = Config()
        self.s3_manager = S3Manager(self.config)
        self.file_manager = FileManager(self.config, self.s3_manager)
        self.doc_processor = DocumentProcessor()
        self.vector_store = SessionVectorStore(self.config)
        self.llm_manager = LLMManager(self.config)
        self.chat_history = []  


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
            logger.error(f"RAG init error: {str(e)}")
            return False

    def retrieve_by_file_id(self, file_id: int, question: str, model_name: Optional[str] = None, strategy: str = "Vector Search"):
        chunks = self.vector_store.load_documents_by_file_id(file_id)
        if not chunks:
            return {
                "preview": [],
                "answer": (x for x in ["No chunks found for this file."]),
                "sources": []
            }

        answer_gen = self.llm_manager.generate_response(question, chunks, model_name=model_name)


        preview = [{"filename": doc.metadata.get("filename", "Unknown"),
                    "file_type": doc.metadata.get("file_type", "Unknown"),
                    "url": doc.metadata.get("source", ""),
                    "excerpt": doc.page_content[:1000] + "..."} for doc in chunks[:3]]

        return {
            "preview": preview,
            "answer": answer_gen,
            "sources": preview
        }



    def index_document(self, file_path: str) -> bool:
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return False

            file_info = self.file_manager.store_file(
                file_path,
                self.vector_store.session_id
            )

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
            logger.error(f"Indexing error: {str(e)}")
            return False

    def list_files(self, session_only: bool = True) -> List[Dict]:
        """List files indexed in the current session or all files."""
        if session_only:
            return self.file_manager.list_files(self.vector_store.session_id)
        else:
            return self.file_manager.list_files()

    def download_file(self, file_id: int, destination_path: str) -> bool:
        """Download a file to the specified path."""
        file_info = self.file_manager.get_file(file_id)
        if not file_info:
            return False

        # If there's a local copy, copy it
        if os.path.exists(file_info["file_path"]):
            try:
                import shutil
                shutil.copy2(file_info["file_path"], destination_path)
                return True
            except Exception as e:
                logger.error(f"Failed to copy file: {str(e)}")

        # If S3 is enabled and we have an object key, download from S3
        if self.config.s3_enabled and "object_key" in file_info:
            return self.s3_manager.download_file(file_info["object_key"], destination_path)

        return False

    def query(self, question: str, model_name: Optional[str] = None, strategy: str = "Vector Search") -> Dict[str, Any]:
        try:
            logger.info(f"Running query with strategy: {strategy}")

            conversation_context = ""
            if self.chat_history:
                for message in self.chat_history:
                    role = message.get("role", "user").capitalize()
                    content = message.get("content", "")
                    conversation_context += f"{role}: {content}\n"


            # --- NEW: Direct LLM answer without context ---
            if strategy == "LLM Only":
                logger.info("Using LLM only; skipping document retrieval.")
                prompt = (
                    "You are a helpful assistant. Please answer the following question as clearly and accurately as possible.\n\n"
                    f"Question: {question}"
                )
                response = self.llm_manager.client.chat.completions.create(
                    model=model_name or self.config.groq_model,
                    messages=[
                        {"role": "system", "content": "Answer the user's question."},
                        {"role": "user", "content": question}
                    ],
                    temperature=0.5,
                    max_tokens=self.config.max_tokens,
                    stream=True
                )
                return {
                    "preview": [],
                    "answer": (chunk.choices[0].delta.content for chunk in response),
                    "sources": []
                }

            # --- Normal RAG flow ---
            docs_with_scores = self.vector_store.similarity_search_with_score(question, k=self.config.search_k)

            if not docs_with_scores:
                logger.warning("No documents retrieved from vector store.")
                return {
                    "preview": [],
                    "answer": (x for x in ["No relevant information found."]),
                    "sources": []
                }

            if strategy == "Vector Search":
                relevant_docs = [doc for doc, _ in docs_with_scores[:self.config.max_context_documents]]
                logger.info(f"Selected {len(relevant_docs)} docs via Vector Search")

            elif strategy == "Semantic Search (LLM Rerank)":
                reranked_docs = self.llm_manager.rank_relevance(question, docs_with_scores[:10])
                relevant_docs = reranked_docs[:self.config.max_context_documents]
                logger.info(f"Selected {len(relevant_docs)} docs via Semantic Search reranking")

            elif strategy == "Hybrid":
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
                logger.info(f"Selected {len(relevant_docs)} docs via Hybrid strategy")

            elif strategy == "BM25":
                relevant_docs = self.vector_store.bm25_search(question, k=self.config.max_context_documents)
                logger.info(f"Selected {len(relevant_docs)} docs via BM25 lexical search")


            else:
                logger.warning(f"Unknown strategy '{strategy}'. Falling back to Vector Search.")
                relevant_docs = [doc for doc, _ in docs_with_scores[:self.config.max_context_documents]]

            # Generate output
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

            full_prompt = f"{conversation_context}\nUser: {question}\nAssistant:"

            answer = self.llm_manager.generate_response(full_prompt, relevant_docs, model_name=model_name)
            sources = [{"url": p["url"], "filename": p["filename"], "file_type": p["file_type"]} for p in preview]
            final_answer = "".join(chunk or "" for chunk in answer)
            self.chat_history.append({"role": "user", "content": question})
            self.chat_history.append({"role": "assistant", "content": final_answer})


            return {
                "preview": preview,
                "answer": answer,
                "sources": sources
            }

        except Exception as e:
            logger.error(f"Query error: {str(e)}", exc_info=True)
            return {
                "preview": preview,
                "answer": iter([final_answer]),  # make it a generator again
                "sources": sources
            }


        
    
# Gradio Interface
# Gradio Interface
def run_gradio_app():
    with gr.Blocks(title="Session-Based Document QA") as demo:
        session_id = gr.State(None)
        chat_history = gr.State([])

        gr.HTML("""
            <style>
                .gradio-container { font-family: 'Segoe UI', sans-serif; }
                .fixed-textbox, .fixed-dropdown, .fixed-button, .fixed-file { width: 100% !important; }
                .preview-box { max-height: 120px; overflow-y: auto; white-space: pre-wrap; margin-top: -24px !important; }
                .json-box { max-height: 100px; overflow-y: auto; white-space: pre-wrap; margin-top: 4px; }
            </style>
        """)

        gr.Markdown("## Session-Based Document QA with Conversational Memory")

        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                file_input = gr.File(label="Upload New File", elem_classes=["fixed-file"])
                past_docs_dd = gr.Dropdown(label="Past Documents", choices=get_past_files(), value=None, interactive=True, allow_custom_value=True, elem_classes=["fixed-dropdown"])
                retrieval_strategy = gr.Dropdown(["Vector Search", "Semantic Search (LLM Rerank)", "Hybrid", "LLM Only", "BM25"], value="Vector Search", label="Retrieval Strategy", elem_classes=["fixed-dropdown"])
                model_choice = gr.Dropdown(["llama-3.1-8b-instant", "deepseek-r1-distill-llama-70b", "gemma2-9b-it"], value="llama-3.1-8b-instant", label="LLM Model", elem_classes=["fixed-dropdown"])

                with gr.Row():
                    delete_btn = gr.Button("Delete Selected File", elem_classes=["fixed-button"])
                    download_btn = gr.Button("Download Selected File", elem_classes=["fixed-button"])

            with gr.Column(scale=3):
                chatbot_ui = gr.Chatbot(label="Conversation", show_label=True, type="messages")
                chat_input = gr.Textbox(label="Ask a Question", placeholder="Ask something about the document...", lines=1, elem_classes=["fixed-textbox"])
                preview_out = gr.Textbox(label="Contextual Preview", lines=2, interactive=False, elem_classes=["fixed-textbox", "preview-box"])
                sources_out = gr.JSON(label="Sources / Citations", elem_classes=["json-box"])
                send_btn = gr.Button("Send", elem_classes=["fixed-button"])

        send_btn.click(
            fn=smart_upload_or_query,
            inputs=[file_input, past_docs_dd, chat_input, model_choice, retrieval_strategy, session_id, chat_history],
            outputs=[preview_out, chatbot_ui, sources_out, past_docs_dd, session_id]
        )

    demo.launch(show_api=False, share=False, inbrowser=True)





# CLI and Main
def run_interactive_mode():
    rag = SessionBasedRAG()
    if not rag.initialize():
        print("Failed to initialize RAG system. Check configuration.")
        return

    print("\n=== Session-Based RAG Interactive Mode ===")
    print("First, let's index some documents. Type 'done' when finished.")

    indexed_files = []
    while True:
        file_path = input("\nEnter document path (or 'done' to continue): ")
        if file_path.lower() == 'done':
            break

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        print(f"Indexing {file_path}...")
        success = rag.index_document(file_path)
        if success:
            print(f"Successfully indexed: {file_path}")
            indexed_files.append(file_path)
        else:
            print(f"Failed to index: {file_path}")

    if not indexed_files:
        print("No documents were indexed. Exiting.")
        return

    print("\n=== Documents indexed successfully ===")
    print("You can now ask questions about your documents. Type 'exit' to quit.")

    while True:
        question = input("\nYour question: ")
        if question.lower() in ['exit', 'quit']:
            break

        print("\nSearching for relevant information...")
        result = rag.query(question)

        print("\n=== Answer ===")
        print(result["answer"])

        if result["sources"]:
            print("\n=== Sources ===")
            for i, source in enumerate(result["sources"]):
                if isinstance(source, dict):
                    print(f"{i+1}. {source.get('filename', 'Unknown')} ({source.get('file_type', 'file')})")
                else:
                    print(f"{i+1}. {source}")

    print("\nExiting interactive mode. Goodbye!")

if __name__ == "__main__":
    run_gradio_app()
