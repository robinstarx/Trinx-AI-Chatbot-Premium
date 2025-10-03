import os
import logging
from typing import Callable, Dict, List
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.schema import Document

# ── LOGGING ─────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

LoaderFactory = Callable[[str], List[Document]]


LOADER_FACTORIES: Dict[str, LoaderFactory] = {
    ".pdf": lambda path: PyPDFLoader(path).load(),
    ".txt": lambda path: TextLoader(path, autodetect_encoding=True).load(),
    ".doc": lambda path: UnstructuredWordDocumentLoader(path).load(),
    ".docx": lambda path: UnstructuredWordDocumentLoader(path).load(),
}


def _load_file(file_path: str) -> List[Document]:
    extension = os.path.splitext(file_path)[1].lower()
    loader_factory = LOADER_FACTORIES.get(extension)

    if not loader_factory:
        logger.warning(f"Unsupported file type: {file_path}")
        return []

    try:
        docs = loader_factory(file_path)
        logger.info(f"Successfully loaded {len(docs)} documents from {file_path}")
        return docs
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}")
        return []


def load_documents(source_path: str) -> List[Document]:
    """
    Load documents from either a single supported file or a folder containing supported files.
    Supported file types: PDF, DOC, DOCX, TXT.
    """
    documents = []
    logger.info(f"Attempting to load documents from: {source_path}")

    try:
        if os.path.isfile(source_path):
            docs = _load_file(source_path)
            documents.extend(docs)
        elif os.path.isdir(source_path):
            # Handle folder containing supported files
            for filename in os.listdir(source_path):
                file_path = os.path.join(source_path, filename)
                if os.path.isfile(file_path):
                    docs = _load_file(file_path)
                    documents.extend(docs)
        else:
            logger.error(f"Path does not exist: {source_path}")
            return []

        logger.info(f"Finished loading documents. Total documents loaded: {len(documents)}")
        return documents

    except FileNotFoundError:
        logger.error(f"Path not found: {source_path}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading documents: {e}")
        return []


def load_single_document(file_path: str) -> List[Document]:
    """Load documents from a single supported file (PDF, DOC, DOCX, TXT)."""
    if not os.path.isfile(file_path):
        logger.error(f"File not found: {file_path}")
        return []

    return _load_file(file_path)


def load_single_pdf(pdf_path: str) -> List[Document]:
    """
    Load documents from a single PDF file.
    """
    if not os.path.isfile(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        return []

    if not pdf_path.lower().endswith(".pdf"):
        logger.warning(f"Expected a PDF file but received: {pdf_path}")
        return []

    return _load_file(pdf_path)


