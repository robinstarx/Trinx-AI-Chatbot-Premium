import os
import logging
from dotenv import load_dotenv

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import tool

# ── LOGGING ─────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# Initialize Serper-based search tool
try:
    load_dotenv()
    os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")
    search = GoogleSerperAPIWrapper(k=10, type="search")
    logger.info("GoogleSerperAPIWrapper initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing GoogleSerperAPIWrapper: {e}")
    raise


@tool
def web_search_tool(query: str) -> str:
    """Up-to-date web info via Serper (Google Search API)"""
    try:
        logger.info(f"Performing web search for query: {query}")
        return search.run(query)
    except Exception as e:
        logger.error(
            f"Error during GoogleSerperAPIWrapper search for query '{query}': {e}"
        )
        return f"WEB_ERROR::{e}"
