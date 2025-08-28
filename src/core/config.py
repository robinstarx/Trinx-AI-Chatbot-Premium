import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

load_dotenv()


class Settings(BaseSettings):
    # API settings
    API_URL: str = "/api/chat-premium"

    # LangSmith settings
    LANGSMITH_ENDPOINT: str = os.getenv("LANGSMITH_ENDPOINT")
    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT")
    SERPER_API_KEY: str = os.getenv("SERPER_API_KEY")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY")

    # LLM settings
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Initialize settings
settings = Settings()
try:
    # Ensure values are in os.environ (important for LangSmith tracing)
    logger.info("Loading environment variables...")
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_ENDPOINT"] = settings.LANGSMITH_ENDPOINT
    os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
    os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT
    os.environ["SERPER_API_KEY"] = settings.SERPER_API_KEY
    os.environ["TAVILY_API_KEY"] = settings.TAVILY_API_KEY
    os.environ["GROQ_API_KEY"] = settings.GROQ_API_KEY
    logger.info("Environment variables loaded successfully.")
except Exception as e:
    logger.error(f"Error loading environment variables: {e}")
    raise