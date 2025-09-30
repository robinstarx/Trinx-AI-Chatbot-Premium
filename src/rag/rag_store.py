import os
import logging
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.tools import tool

from src.utils.document_loader import load_single_document
from src.core.config import Settings

logger = logging.getLogger(__name__)
settings = Settings()
openai_api_key = settings.OPENAI_API_KEY
SAVE_PATH = "vector_store_db"
EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)
pinecone_api_key = settings.PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)

index_name = "langchain-test-index"
pc.describe_index(index_name)
index = pc.Index(index_name)

vector_store = PineconeVectorStore(index=index, embedding=EMBEDDINGS)

# def initialize_pinecone():
#     """Initialize Pinecone index and vector store."""
#     global vector_store
#     try:
#         pc.describe_index(index_name)
#         logger.info(f"Pinecone index {index_name} already exists.")
#     except Exception as e:
#         pc.create_index(
#             name=index_name,
#             dimension=1536,
#             metric="cosine",
#             spec=ServerlessSpec(cloud="aws", region="us-east-1"),
#         )
#         logger.info(f"Created Pinecone index: {index_name}")

#     index = pc.Index(index_name)
#     vector_store = PineconeVectorStore(index=index, embedding=EMBEDDINGS)
#     return vector_store



def add_docs(file_path, vector_store, metadata: dict | None = None):
    """
    Add a list of langchain Document objects to the Pinecone vector store.
    """
    docs = load_single_document(file_path)

    if not docs:
        logger.warning("add_docs called with empty docs; nothing to do.")
        return False

    # if vector_store is None:
    #     vector_store = ""

    #     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    #     chunks = splitter.split_documents(docs)

    #     # attach metadata
    #     if metadata:
    #         for c in chunks:
    #             if c.metadata:
    #                 c.metadata.update(metadata)
    #             else:
    #                 c.metadata = metadata.copy()
    #     vector_store.add_documents(chunks)
    # else:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = splitter.split_documents(docs)



    # attach metadata
    if metadata:
        for c in chunks:
            if c.metadata:
                c.metadata.update(metadata)
            else:
                c.metadata = metadata.copy()
    
    # Add documents to vector store
    vector_store.add_documents(chunks)
    logger.info(f"Added {len(chunks)} chunks to Pinecone index {index_name}.")
    return True if chunks else False

# Preload Trinity docs if they exist (global shared docs)
pdf_path = os.path.join("data", "SRS_trinity_project.pdf")
if os.path.exists(pdf_path):
    try:
        logger.info("Found Trinity PDF -> loading and indexing as global 'trinity' source.")
        # add_docs will create the index if needed
        add_docs(pdf_path, vector_store, {"source": "trinity", "filename": os.path.basename(pdf_path)})
    except Exception as e:
        logger.exception("Failed to preload Trinity docs: %s", e)

@tool
def rag_search_tool(query: str, user_id: str = None, source: str = None) -> str:
    """
    Unified RAG search tool. Provide either user_id (to search user uploads),
    or source="trinity" (to search only Trinity global docs), or neither (search all).
    Returns concatenated page_content strings (or "No relevant documents found.").
    """
    global vector_store

    if vector_store is None:
        logger.info("Vector DB empty: no documents indexed yet.")
        return "No relevant documents found."

    try:
        # Build filter dictionary for metadata-based filtering
        filter_dict = {}
        
        if source == "trinity":
            filter_dict = {"source": "trinity"}
            k = 3  # Fewer results for Trinity docs
            logger.info(f"Searching Trinity docs with query: {query}")
        elif source == "upload":
            filter_dict = {"source": "upload"}
            # If user_id is provided, also filter by user_id
            if user_id:
                filter_dict["user_id"] = user_id
            k = 5  # More results for user uploads
            logger.info(f"Searching uploaded docs with query: {query}, user_id: {user_id}")
        else:
            # Search all documents if no source specified
            k = 5
            logger.info(f"Searching all documents with query: {query}")

        # Perform similarity search with or without filters
        if filter_dict:
            docs = vector_store.similarity_search(query, k=k, filter=filter_dict)
        else:
            docs = vector_store.similarity_search(query, k=k)

        if not docs:
            logger.info("No relevant documents found for query: %s", query)
            return "No relevant documents found."

        # Return chunks with attribution for debugging
        result = "\n\n".join(
            f"[{d.metadata.get('filename', 'unknown')} | source={d.metadata.get('source', 'n/a')}]: {d.page_content}"
            for d in docs
        )
        
        logger.info(f"Found {len(docs)} relevant documents")
        return result
        
    except Exception as e:
        logger.exception("Error during RAG search: %s", e)
        return f"RAG_ERROR::{e}"
