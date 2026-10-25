import pytest
from fastapi.testclient import TestClient
from app import app
from evaluate import ProductionEvaluator
from database import init_db, log_inference, get_logs

client = TestClient(app)

def setup_module():
    init_db()

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "device" in data

def test_chair_normalization_and_semantics():
    evaluator = ProductionEvaluator()
    
    # Test synonym mapping & compound words
    caption = "A player eating a hot dog next to a bicycle."
    extracted = evaluator.normalize_and_extract_objects(caption)
    
    assert "person" in extracted
    assert "hot dog" in extracted
    assert "dog" not in extracted # Prevent false positive animal dog
    assert "bicycle" in extracted

    # Test evaluation without ground truth (missing-value semantics -> null, not zero)
    res_no_gt = evaluator.evaluate_caption(caption, ground_truth_objects=None)
    assert res_no_gt["evaluation_status"] == "not_evaluated"
    assert res_no_gt["chair_s"] is None
    assert res_no_gt["chair_i"] is None

    # Test evaluation with correct ground truth
    res_with_gt = evaluator.evaluate_caption(caption, ground_truth_objects=["person", "hot dog", "bicycle"])
    assert res_with_gt["evaluation_status"] == "evaluated"
    assert res_with_gt["chair_s"] == 0.0
    assert res_with_gt["chair_i"] == 0.0

    # Test evaluation with hallucination
    res_hallucination = evaluator.evaluate_caption(caption, ground_truth_objects=["bicycle"])
    assert res_hallucination["evaluation_status"] == "evaluated"
    assert res_hallucination["chair_s"] == 1.0
    assert res_hallucination["chair_i"] > 0.0

def test_logs_endpoint_and_database():
    log_inference(
        image_id="test_img_001",
        caption="A test caption",
        evaluation_status="not_evaluated",
        chair_s=None,
        chair_i=None
    )
    logs = get_logs(limit=5)
    assert isinstance(logs, list)
    assert len(logs) > 0
    assert logs[0]["image_id"] == "test_img_001"
    assert logs[0]["chair_s"] is None