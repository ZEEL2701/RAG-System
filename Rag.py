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

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("enhanced_rag")

import psycopg2
from psycopg2.extras import RealDictCursor
def get_past_files(session_id: Optional[str] = None) -> List[str]:
    from psycopg2.extras import RealDictCursor
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

            # Filter out missing files
            choices = []
            for r in rows:
                if os.path.exists(r["file_path"]):
                    choices.append(f"{r['file_name']} (ID: {r['id']})")
                else:
                    logger.warning(f"Skipping missing file from dropdown: {r['file_path']}")

            logger.info(f"[Startup] Returning {len(choices)} valid file(s) for dropdown.")
            return choices
    except Exception as e:
        logger.error(f"[Startup] Error during file fetch: {e}")
        return []
    finally:
        conn.close()

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
        conn = psycopg2.connect(self.config.connection_string)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM file_registry WHERE file_name = %s AND session_id = %s",
            (file_name, session_id)
        )
        if cursor.fetchone():
            logger.info(f"File already registered: {file_name}")
            return None

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
        self.session_id = str(uuid.uuid4())
        self.session_docs = []

    def initialize(self):
        try:
            logger.info(f"Initializing embeddings using model {self.config.embedding_model}")

            # update with hugging face embeddings
            self.embeddings = HuggingFaceEmbeddings("sentence-transformers/all-MiniLM-L6-v2")

            logger.info("Embeddings initialized successfully.")

            session_collection = f"{self.config.collection_name}-{self.session_id}"
            logger.info(f"Using session-specific collection: {session_collection}")
            try:
                self.vectorstore = PGVector.from_documents(
                    documents=[],
                    embedding=self.embeddings,
                    collection_name=session_collection,
                    connection_string=self.config.connection_string,
                )
            except Exception as e:
                logger.error(f"PGVector init failed: {e}")
                raise

        except Exception as e:
            logger.error(f"Vector store init error: {str(e)}")
            raise

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

    def semantic_search(self, query: str, k: int = 4) -> List[Document]:
        logger.info("Running SEMANTIC search")
        scored_docs = self.similarity_search_with_score(query , k)
        return[doc for doc, _ in scored_docs]

    def hybrid_search(self, query: str, k: int = 4) -> List[Document]:
        logger.info("Running HYBRID search")
        docs = self.similarity_search(query, k)
        scored = self.similarity_search_with_score(query, k)
        return list({doc.metadata.get("source", id(doc)): doc for doc, _ in scored + [(d, None) for d in docs]}.values())    

    def get_session_document_count(self) -> int:
        """Return the number of documents in the current session."""
        return len(self.session_docs)
    
    def vector_search(self, query: str, k: int = 4) -> List[Document]:
        logger.info("Running VECTOR search")
        return self.similarity_search(query, k)
    

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

    def generate_response(self, query: str, context_docs: List[Document], model_name: Optional[str] = None) -> str:
        try:
            print("------", self.client)
            if not self.client:
                print("---- CLIENT GOING TO INITIALISED ------")
                self.initialize()
                print("------", self.client)

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
                "You are a helpful and precise assistant that provides accurate information based on the context provided. "
                "Follow these rules when generating answers:\n"
                "1. Only use information from the provided context documents.\n"
                "2. If the context doesn't contain information needed to fully answer the question, say so clearly.\n"
                "3. Do not invent or assume information that's not in the context.\n"
                "4. Cite the specific document sources when providing information.\n"
                "5. Format your answer for clarity and readability.\n"
                "6. If different documents contain conflicting information, acknowledge this and explain the differences."
            )

            user_prompt = (
                f"Question: {query}\n\n"
                f"Context Documents:\n{context}\n\n"
                f"Please provide a comprehensive answer based solely on the provided context documents. "
                f"Cite specific documents when providing information."
            )

        
            chosen_model = model_name or self.config.groq_model

            response = self.client.chat.completions.create(
                model=chosen_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=8192,
            )

            answer = response.choices[0].message.content
            logger.info(f"Generated response for query: {query[:50]}...")
            return answer
        except Exception as e:
            logger.error(f"Response error: {str(e)}")
            return "An error occurred while generating the answer."

    def rank_relevance(self, query: str, docs_with_scores: List[tuple]) -> List[Document]:
        """Rank documents by relevance and filter out irrelevant ones."""
        try:
            if not self.client:
                self.initialize()
            if not docs_with_scores:
                return []

            threshold_score = 0.3  # Adjust as needed
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

    def index_document(self, file_path: str) -> bool:
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}, skipping index.")
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

    def list_files(self, session_only=False, session_id=None):
        return self.file_manager.list_files(session_id=session_id if session_only else None)

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

    def query(self, question: str, model_name: Optional[str] = None, strategy: str = "vector") -> Dict[str, Any]:
        try:
            logger.info(f"Running query with strategy: {strategy}")

            # LLM Only mode: answer without any documents
            if strategy == "LLM Only":
                logger.info("Using LLM Only mode — no document context.")
                answer = self.llm_manager.generate_response(question, [], model_name=model_name)
                return {
                    "preview": [],
                    "answer": answer,
                    "sources": []
                }

            # Otherwise use RAG (Vector, Semantic, or Hybrid)
            docs_with_scores = self.vector_store.similarity_search_with_score(question, k=self.config.search_k)

            if not docs_with_scores:
                logger.warning("No documents retrieved from vector store.")
                return {
                    "preview": [],
                    "answer": "No relevant information found in the uploaded document.",
                    "sources": []
                }

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
                return {
                    "preview": [],
                    "answer": f"Unknown search type '{strategy}'",
                    "sources": []
                }

            # Build the preview
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
            return {"preview": preview, "answer": answer, "sources": sources}

        except Exception as e:
            logger.error(f"Query error: {str(e)}")
            return {
                "preview": [],
                "answer": "An error occurred during the query process.",
                "sources": []
            }

# Gradio Interface
def run_gradio_app():
    rag_instances = {}

    def create_rag_instance():
        session_id = str(uuid.uuid4())
        rag = SessionBasedRAG()
        if not rag.initialize():
            logger.error("Failed to initialize RAG system. Check configuration.")
            return None
        rag_instances[session_id] = rag
        return rag

    def get_dropdown_files(session_id=None):
        # Use the new get_past_files() function
        files = get_past_files()
        dropdown_update = gr.update(choices=files, value=None)
        return dropdown_update, session_id

    def upload_and_query(file, user_query, model_name, search_type, selected_file_dropdown, session_id=None):
        if not session_id or session_id not in rag_instances:
            rag = create_rag_instance()
            if not rag:
                return "Failed to initialize the system.", "", [], [], [], session_id
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

        # Case 1: New file upload
        if file:
            success = rag.index_document(file.name)
            if not success:
                updated_choices = get_past_files()
                return "Failed to index the document.", "", [], gr.update(choices=updated_choices, value=None), session_id

        # Case 2: Selected file from dropdown
        elif selected_file_dropdown:
            try:
                file_id = int(selected_file_dropdown.split("ID: ")[1].rstrip(")"))
            except Exception:
                updated_choices = get_past_files()
                return "Invalid file selection format.", "", [], gr.update(choices=updated_choices, value=None), session_id

            file_info = rag.file_manager.get_file(file_id)
            if not file_info:
                updated_choices = get_past_files()
                return "Selected file not found.", "", [], gr.update(choices=updated_choices, value=None), session_id

            file_path = file_info["file_path"]
            
            # Modified approach: Process the document directly without re-storing it
            try:
                documents = rag.doc_processor.load_file(file_path, rag.config)
                for doc in documents:
                    doc.metadata["file_id"] = file_info["file_id"]
                    doc.metadata["source"] = file_info.get("download_url", file_path)
                rag.vector_store.add_documents(documents)
                logger.info(f"Previously indexed document loaded: {file_path}")
                success = True
            except Exception as e:
                logger.error(f"Error processing previously indexed document: {str(e)}")
                success = False
                
            if not success:
                updated_choices = get_past_files()
                return "Failed to load selected file.", "", [], gr.update(choices=updated_choices, value=None), session_id

        if not user_query.strip():
            updated_choices = get_past_files()
            return "Please enter a question.", "", [], [], [], session_id

        if rag.vector_store.get_session_document_count() == 0:
            updated_choices = get_past_files()
            return "No content found in the document.", "", [], gr.update(choices=updated_choices, value=None), session_id

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
            return "No relevant information found.", "", [], [], [], session_id

        answer = rag.llm_manager.generate_response(user_query, relevant_docs, model_name=model_name)

        sources = [
            {
                "label": f"{doc.metadata.get('filename', 'Unknown')} (ID: {doc.metadata.get('file_id', '-')})",
                "file_id": doc.metadata.get("file_id", None),
                "file_type": doc.metadata.get("file_type", "file"),
                "citation": doc.page_content[:200].strip() + ("..." if len(doc.page_content) > 200 else ""),
                "url": doc.metadata.get("source", "")
            }
            for doc in relevant_docs
        ]

        updated_choices = get_past_files()
        return "", answer, sources, gr.update(choices=updated_choices, value=None), session_id

    with gr.Blocks(css=".gr-block { font-family: 'Segoe UI', sans-serif; padding: 8px; }") as demo:
        session_id = gr.State(None)

        gr.Markdown("### Document QA Application")

        with gr.Row():
            # LEFT: Upload + Past Files + Controls
            with gr.Column(scale=1):
                file_input = gr.File(label="Upload Document", file_types=["file"], scale=1)
                file_dropdown = gr.Dropdown(label="Previously Uploaded File", choices=[], interactive=True, allow_custom_value=True, scale=1)
                model_selector = gr.Dropdown(
                    choices=["llama-3.1-8b-instant", "deepseek-r1-distill-llama-70b", "gemma2-9b-it"],
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

            # CENTER: Answer → Question → Submit
            with gr.Column(scale=2):
                answer_output = gr.Textbox(label="Answer", lines=15, interactive=False)
                question_input = gr.Textbox(label="Question", placeholder="Ask a question...")
                process_button = gr.Button("Submit")

            # RIGHT: Sources
            with gr.Column(scale=1):
                sources_output = gr.JSON(label="Sources with Context")
        def download_selected_file(selected_file_dropdown, session_id=None):
            if not selected_file_dropdown:
                return "No file selected."

            try:
                file_id = int(selected_file_dropdown.split("ID: ")[1].rstrip(")"))
            except Exception:
                return "Invalid file selection format."

            rag = rag_instances.get(session_id)
            if not rag:
                return "Session not found."

            file_info = rag.file_manager.get_file(file_id)
            if not file_info:
                return "File not found."

            if file_info.get("download_url"):
                return file_info["download_url"]
            else:
                return "No download URL available."

        def delete_selected_file(selected_file_dropdown, session_id=None):
            if not selected_file_dropdown:
                return "No file selected."

            try:
                file_id = int(selected_file_dropdown.split("ID: ")[1].rstrip(")"))
            except Exception:
                return "Invalid file selection format."

            rag = rag_instances.get(session_id)
            if not rag:
                return "Session not found."

            success = rag.file_manager.delete_file(file_id)
            if success:
                return "File deleted successfully."
            else:
                return "Failed to delete file."                

        demo.load(
            fn=get_dropdown_files,
            inputs=[session_id],
            outputs=[file_dropdown, session_id]
        )    
        process_button.click(
            upload_and_query,
            inputs=[file_input, question_input, model_selector, search_selector, file_dropdown, session_id],
            outputs=[answer_output, answer_output, sources_output, file_dropdown, session_id]
        )
        get_download_button.click(
            download_selected_file,
            inputs=[file_dropdown, session_id],
            outputs=[download_link_output]
        )

        delete_button.click(
            delete_selected_file,
            inputs=[file_dropdown, session_id],
            outputs=[delete_output]
        )       

    demo.launch()



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

def main():
    parser = argparse.ArgumentParser(description="Session-Based RAG System")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    subparsers.add_parser("setup", help="Create a sample .env file")

    index_parser = subparsers.add_parser("index", help="Index a document")
    index_parser.add_argument("file_path", type=str, help="Path to document")

    query_parser = subparsers.add_parser("query", help="Query the system")
    query_parser.add_argument("question", type=str, help="Query question")

    subparsers.add_parser("interactive", help="Start interactive mode")
    subparsers.add_parser("serve", help="Run Gradio UI (Upload & Query in one step)")

    file_parser = subparsers.add_parser("files", help="List and manage files")
    file_parser.add_argument("--list", action="store_true", help="List all files")
    file_parser.add_argument("--download", type=int, help="Download file by ID")
    file_parser.add_argument("--output", type=str, help="Output path for download")
    file_parser.add_argument("--delete", type=int, help="Delete file by ID")

    args = parser.parse_args()

    if args.command == "setup":
        if not os.path.exists(".env"):
            with open(".env", "w") as f:
                f.write("""# Database configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=ragdb
DB_USER=postgres
DB_PASSWORD=your_password

# Ollama configuration
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text

# Document processing
CHUNK_SIZE=500
CHUNK_OVERLAP=50
COLLECTION_NAME=rag-pgvector

# RAG configuration
MAX_CONTEXT_DOCUMENTS=7
SEARCH_K=10

# Groq API configuration
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.1-8b-instant

# AWS S3 configuration
S3_ENABLED=False
S3_BUCKET_NAME=your-bucket
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1
""")
            logger.info("Sample .env file created. Edit it with your configuration.")
        return
    elif args.command == "index":
        rag = SessionBasedRAG()
        if rag.initialize():
            if rag.index_document(args.file_path):
                print(f"Successfully indexed: {args.file_path}")
            else:
                print(f"Failed to index: {args.file_path}")
        else:
            print("Failed to initialize RAG system")
    elif args.command == "query":
        rag = SessionBasedRAG()
        if rag.initialize():
            result = rag.query(args.question)
            print("\nAnswer:", result["answer"])
            if result["sources"]:
                print("\nSources:")
                for i, source in enumerate(result["sources"]):
                    if isinstance(source, dict):
                        print(f"{i+1}. {source.get('filename', 'Unknown')} ({source.get('file_type', 'file')})")
                    else:
                        print(f"{i+1}. {source}")
        else:
            print("Failed to initialize RAG system")

    elif args.command == "files":
        rag = SessionBasedRAG()
        if not rag.initialize():
            print("Failed to initialize RAG system")
            return

        if args.list:
            files = rag.list_files(session_only=False)
            if not files:
                print("No files found.")
            else:
                print(f"{'ID':<5} {'Name':<30} {'Type':<10} {'Uploaded':<20}")
                print("-" * 65)
                for f in files:
                    print(f"{f['file_id']:<5} {f['file_name']:<30} {f['file_type']:<10} {f['upload_date']:<20}")

        elif args.download and args.output:
            if rag.download_file(args.download, args.output):
                print(f"File downloaded to {args.output}")
            else:
                print("Failed to download file")

        elif args.delete:
            if rag.file_manager.delete_file(args.delete):
                print(f"File with ID {args.delete} deleted")
            else:
                print(f"Failed to delete file with ID {args.delete}")

        else:
            print("Use --list to list files, --download ID --output PATH to download, or --delete ID to delete")
    elif args.command == "interactive":
        run_interactive_mode()
    elif args.command == "serve":
        run_gradio_app()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()