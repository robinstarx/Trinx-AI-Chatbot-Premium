import pytest
import os
import time
from src.core.config import Settings


@pytest.fixture(scope="session", autouse=True)
def setup_langsmith_for_tests():
    """Configure LangSmith for testing environment."""
    settings = Settings()

    # Ensure all LangSmith environment variables are properly set
    os.environ["LANGSMITH_TRACING"] = "true"

    if settings.LANGSMITH_API_KEY:
        os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
    if settings.LANGSMITH_ENDPOINT:
        os.environ["LANGSMITH_ENDPOINT"] = settings.LANGSMITH_ENDPOINT
    if settings.LANGSMITH_PROJECT:
        # Use a separate project for tests to avoid mixing with production traces
        os.environ["LANGSMITH_PROJECT"] = f"{settings.LANGSMITH_PROJECT}-test"

    yield

    # Give time for any pending traces to complete before test session ends
    time.sleep(3)


@pytest.fixture(autouse=True)
def wait_for_trace_completion():
    """Ensure traces have time to complete after each test."""
    yield
    # Wait for trace to complete after each test
    time.sleep(2)
