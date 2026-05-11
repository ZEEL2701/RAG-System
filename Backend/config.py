import os
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger("enhanced_rag")

class Config:
    """Configuration for the RAG system."""

    def __init__(self):
        self.db_host = os.getenv("DB_HOST", "localhost")
        self.db_port = os.getenv("DB_PORT", "5432")
        self.db_name = os.getenv("DB_NAME", "Your database name")
        self.db_user = os.getenv("DB_USER", "Your Username")
        self.db_password = os.getenv("DB_PASSWORD", "Your Password")

        # Ollama configuration
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

        # Document processing
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))
        self.collection_name = os.getenv("COLLECTION_NAME", "rag-pgvector")

        self.max_tokens = int(os.getenv("MAX_TOKENS", "256"))

        # Groq API configuration
        self.groq_base_url = "https://api.groq.com"
        self.groq_api_key = os.getenv("GROQ_API_KEY", "your_groq_api_key_here")
        self.groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.groq_completion_max_tokens = int(os.getenv("GROQ_COMPLETION_MAX_TOKENS", "512"))
        self.max_chars_per_context_doc = int(os.getenv("MAX_CHARS_PER_CONTEXT_DOC", "900"))

        # RAG configuration
        self.max_context_documents = int(os.getenv("MAX_CONTEXT_DOCUMENTS", "3"))
        self.search_k = int(os.getenv("SEARCH_K", "5"))

        # AWS S3 configuration
        self.s3_enabled = os.getenv("S3_ENABLED", "False").lower() == "true"
        self.s3_bucket_name = os.getenv("S3_BUCKET_NAME", "")
        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID", "")
        self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", "")
        self.aws_region = os.getenv("AWS_REGION", "")

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
