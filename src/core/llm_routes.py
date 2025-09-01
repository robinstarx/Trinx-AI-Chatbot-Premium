import logging
from langchain_groq import ChatGroq
from src.core.state import QueryDecision, FetchCoinPriceDecision
from src.core.config import Settings

# ── LOGGING ────────
logger = logging.getLogger(__name__)
settings = Settings()

# ── LLM instances with structured output where needed ───────────────
try:
    logger.info("Initializing LLM instances...")
    query_router_llm = ChatGroq(
        model=settings.GROQ_MODEL, temperature=0
    ).with_structured_output(QueryDecision)
    coin_symbol_converter_llm = ChatGroq(
        model=settings.GROQ_MODEL, temperature=0, max_tokens=10
    ).with_structured_output(FetchCoinPriceDecision)
    answer_compose_llm = ChatGroq(model=settings.GROQ_MODEL, temperature=0.7)
    trinity_coin_details_llm = ChatGroq(model=settings.GROQ_MODEL, temperature=0.7)
    logger.info("LLM instances initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing LLM instances: {e}")
    raise
