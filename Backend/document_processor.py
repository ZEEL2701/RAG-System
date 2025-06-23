import os
import pandas as pd
import logging
from typing import List
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

logger = logging.getLogger("enhanced_rag")

class DocumentProcessor:
    @staticmethod
    def load_file(file_path: str, config) -> List[Document]:
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
                doc.metadata["file_type"] = ext[1:]  # remove dot

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
    def split_documents(documents: List[Document], config) -> List[Document]:
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
