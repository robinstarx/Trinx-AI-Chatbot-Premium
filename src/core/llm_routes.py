import os
import logging
from langchain_groq import ChatGroq
from src.core.state import QueryDecision, FetchCoinPriceDecision
from dotenv import load_dotenv

# ── LOGGING ────────
logger = logging.getLogger(__name__)

# ── Load environment variables ──────
try:
    load_dotenv()
    os.environ["LANGSMITH_TRACING"] = 'true'
    os.environ["LANGSMITH_ENDPOINT"] = os.getenv('LANGSMITH_ENDPOINT')
    os.environ["LANGSMITH_API_KEY"] = os.getenv('LANGSMITH_API_KEY')
    os.environ["LANGSMITH_PROJECT"] = os.getenv('LANGSMITH_PROJECT')
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    logger.info("Environment variables loaded successfully.")
except Exception as e:
    logger.error(f"Error loading environment variables: {e}")
    raise

# ── LLM instances with structured output where needed ───────────────
try:
    logger.info("Initializing LLM instances...")
    query_router_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0,api_key=os.getenv("GROQ_API_KEY")).with_structured_output(QueryDecision)
    coin_symbol_converter_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, max_tokens=10,api_key=os.getenv("GROQ_API_KEY")).with_structured_output(FetchCoinPriceDecision)
    answer_compose_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7,api_key=os.getenv("GROQ_API_KEY"))
    trinity_coin_details_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7,api_key=os.getenv("GROQ_API_KEY"))
    logger.info("LLM instances initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing LLM instances: {e}")
    raise