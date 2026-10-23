import pytest
from fastapi.testclient import TestClient
from app import app
from evaluate import ProductionEvaluator

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

def test_chair_normalization_logic():
    evaluator = ProductionEvaluator()
    # Test synonym mapping ("player" -> "person") and compound handling ("hot dog")
    caption = "A player eating a hot dog near a bicycle."
    extracted = evaluator.normalize_and_extract_objects(caption)
    
    assert "person" in extracted  # from 'player'
    assert "hot dog" in extracted # compound preserved
    assert "dog" not in extracted # animal dog not falsely triggered
    assert "bicycle" in extracted

def test_caption_endpoint_validation():
    response = client.post("/caption")
    assert response.status_code == 422