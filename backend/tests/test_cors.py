from fastapi.testclient import TestClient
from backend.main import app
import os

client = TestClient(app)

def test_cors_headers():
    # Get the frontend URL from the environment variable for the test
    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
    response = client.get("/", headers={"Origin": frontend_url})
    assert "access-control-allow-origin" in response.headers
    assert response.headers["access-control-allow-origin"] == frontend_url

def test_cors_headers_fails_with_wrong_origin():
    response = client.get("/", headers={"Origin": "http://wrong-origin:3000"})
    assert "access-control-allow-origin" not in response.headers