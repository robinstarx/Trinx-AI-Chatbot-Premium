import os
import logging
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from dotenv import load_dotenv

# ── LOGGING ─────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# Initialize Tavily search tool
# Note: Ensure you have the correct API key set in your environment for Tavily
try:
    load_dotenv()
    os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
    tavily = TavilySearch(max_results=5, search_depth="advanced", include_answer="advanced")
    logger.info("TavilySearch tool initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing TavilySearch tool: {e}")
    raise

@tool
def web_search_tool(query: str) -> str:
    """Up-to-date web info via Tavily"""
    try:
        logger.info(f"Performing web search for query: {query}")
        result = tavily.invoke({"query": query})
        answer = result.get("answer")
        if answer:
            logger.info(f"Web search produced an answer for query: {result.get('query')}")
            return answer
        detail = result.get("detail")
        if isinstance(detail, dict) and "error" in detail:
            logger.warning(f"Web search returned error for query '{result.get('query')}': {detail.get('error')}")
            return str(detail.get("error"))
        return str(result)
    except Exception as e:
        logger.error(f"Error during web search for query '{query}': {e}")
        return f"WEB_ERROR::{e}"