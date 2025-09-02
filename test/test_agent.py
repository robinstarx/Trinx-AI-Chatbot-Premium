from fastapi.testclient import TestClient
from src.core.config import Settings
from src.api.main import app

settings = Settings()

client = TestClient(app)


def test_api_health():
    """
    Tests the /health endpoint.

    Verifies that the endpoint returns a 200 OK status and the expected
    JSON response: `{"status": "healthy"}`.
    """
    response = client.get("/check-health")
    assert response.status_code == 200
    assert response.json() == {"status": "I'm alive and healthy"}


def test_hr_qa_service_route():
    data = {
        "session_id": "1001",
        "query": "what is the pirce of the BTC?",
    }
    response = client.post(settings.API_URL, json=data)
    assert response.status_code == 200

    content = response.json().get("response")
    assert content is not None
    assert isinstance(content, str)
