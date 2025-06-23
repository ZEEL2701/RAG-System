import uuid
import logging
from typing import List, Optional
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import PGVector

logger = logging.getLogger("enhanced_rag")

class HuggingFaceEmbeddings:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=False)

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text])[0]

class SessionVectorStore:
    def __init__(self, config):
        self.config = config
        self.embeddings = None
        self.vectorstore = None
        self.session_id = str(uuid.uuid4())
        self.session_docs: List[Document] = []

    def initialize(self):
        try:
            logger.info(f"Initializing embeddings using model {self.config.embedding_model}")
            self.embeddings = HuggingFaceEmbeddings(self.config.embedding_model)
            logger.info("Embeddings initialized successfully.")

            session_collection = f"{self.config.collection_name}-{self.session_id}"
            logger.info(f"Using session-specific collection: {session_collection}")
            self.vectorstore = PGVector.from_documents(
                documents=[],
                embedding=self.embeddings,
                collection_name=session_collection,
                connection_string=self.config.connection_string,
            )
        except Exception as e:
            logger.error(f"Vector store init error: {str(e)}")
            raise

    def add_documents(self, documents: List[Document]):
        if not documents:
            logger.warning("No documents to add.")
            return
        if not self.embeddings:
            self.initialize()

        for doc in documents:
            doc.metadata["session_id"] = self.session_id

        try:
            self.vectorstore.add_documents(documents)
            self.session_docs.extend(documents)
            logger.info(f"Added {len(documents)} docs to session store. Total: {len(self.session_docs)}")
        except Exception as e:
            logger.error(f"Error adding docs: {str(e)}")
            raise

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        if not self.vectorstore:
            logger.warning("Vector store not initialized.")
            return []
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            logger.info(f"Found {len(docs)} docs for query: {query[:50]}...")
            return docs
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return []

    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        if not self.vectorstore:
            logger.warning("Vector store not initialized.")
            return []
        try:
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
            logger.info(f"Found {len(docs_with_scores)} docs with scores for query: {query[:50]}...")
            return docs_with_scores
        except Exception as e:
            logger.error(f"Search with score error: {str(e)}")
            return []

    def semantic_search(self, query: str, k: int = 4) -> List[Document]:
        logger.info("Running SEMANTIC search")
        scored_docs = self.similarity_search_with_score(query, k)
        return [doc for doc, _ in scored_docs]

    def hybrid_search(self, query: str, k: int = 4) -> List[Document]:
        logger.info("Running HYBRID search")
        docs = self.similarity_search(query, k)
        scored = self.similarity_search_with_score(query, k)
        combined = scored + [(d, None) for d in docs]
        unique_docs = {}
        for doc, _ in combined:
            uid = (doc.metadata.get("source", id(doc)), doc.metadata.get("chunk", ""))
            if uid not in unique_docs:
                unique_docs[uid] = doc
        return list(unique_docs.values())

    def get_session_document_count(self) -> int:
        return len(self.session_docs)

    def vector_search(self, query: str, k: int = 4) -> List[Document]:
        logger.info("Running VECTOR search")
        return self.similarity_search(query, k)
