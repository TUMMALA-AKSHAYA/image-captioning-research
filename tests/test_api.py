import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "device" in data

def test_logs_endpoint():
    response = client.get("/logs?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert isinstance(data["logs"], list)

def test_caption_endpoint_validation():
    # Sending request without file should return 422 validation error
    response = client.post("/caption")
    assert response.status_code == 422