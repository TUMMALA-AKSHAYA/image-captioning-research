import io
import json
import sqlite3

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app as api
import database
from dataset_evaluation import DatasetEvaluator
from evaluate import ProductionEvaluator
from caption_metrics import CaptionMetricsEvaluator


class FakePipeline:
    device = "cpu"

    def generate_caption(self, image_bytes):
        return {
            "caption": "A person riding a bicycle.",
            "word_count": 5,
            "latency_ms": 12.5,
            "device": "cpu",
            "grounding_boxes": [
                {
                    "label": "bicycle",
                    "confidence": 0.91,
                    "x": 10.0,
                    "y": 20.0,
                    "w": 100.0,
                    "h": 80.0,
                    "color": "#10b981",
                }
            ],
        }


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_NAME", str(tmp_path / "test.db"))
    monkeypatch.setattr(api, "ProductionCaptionPipeline", FakePipeline)
    with TestClient(api.app) as test_client:
        yield test_client


def encoded_image(format="JPEG"):
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color="white").save(buffer, format=format)
    return buffer.getvalue()


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "online", "device": "cpu"}


def test_caption_endpoint_returns_detection_and_logs_without_claiming_evaluation(client):
    response = client.post(
        "/caption",
        files={"file": ("image.jpg", encoded_image(), "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["caption"] == "A person riding a bicycle."
    assert data["grounding_boxes"][0]["label"] == "bicycle"

    logs = client.get("/logs").json()["logs"]
    assert logs[0]["caption"] == data["caption"]
    assert logs[0]["evaluation_status"] == "not_evaluated"
    assert logs[0]["chair_s"] is None


def test_caption_endpoint_rejects_missing_unsupported_and_invalid_images(client):
    assert client.post("/caption").status_code == 422
    unsupported = client.post(
        "/caption", files={"file": ("image.gif", b"not-image", "image/gif")}
    )
    assert unsupported.status_code == 415
    invalid = client.post(
        "/caption", files={"file": ("image.jpg", b"not-image", "image/jpeg")}
    )
    assert invalid.status_code == 400


def make_rule_evaluator():
    evaluator = object.__new__(ProductionEvaluator)
    evaluator.coco_objects = {"person", "bicycle", "dog", "hot dog"}
    evaluator.synonyms = {"player": "person", "bike": "bicycle"}
    return evaluator


def test_chair_inspired_evaluation_handles_missing_and_present_ground_truth():
    evaluator = make_rule_evaluator()
    caption = "A player eating a hot dog next to a bicycle."
    assert evaluator.normalize_and_extract_objects(caption) == ["hot dog", "person", "bicycle"]
    missing = evaluator.evaluate_caption(caption)
    assert missing["evaluation_status"] == "not_evaluated"
    assert missing["chair_s"] is None
    correct = evaluator.evaluate_caption(caption, ["person", "hot dog", "bicycle"])
    assert correct["chair_s"] == 0.0
    assert correct["chair_i"] == 0.0
    hallucinated = evaluator.evaluate_caption(caption, ["bicycle"])
    assert hallucinated["chair_s"] == 1.0
    assert hallucinated["chair_i"] > 0.0


class FakeEvaluator:
    def generate_caption(self, image_bytes):
        return "A person riding a bicycle."

    def evaluate_caption(self, caption, ground_truth_objects=None):
        return make_rule_evaluator().evaluate_caption(caption, ground_truth_objects)


def test_dataset_evaluation_generates_captions_and_skips_unannotated_metrics(tmp_path):
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(encoded_image("PNG"))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({
            "image_path": "sample.png",
            "ground_truth_objects": ["person", "bicycle"],
        }) + "\n" +
        json.dumps({"image_path": "sample.png"}) + "\n",
        encoding="utf-8",
    )

    evaluator = DatasetEvaluator(FakeEvaluator())
    result = evaluator.evaluate_dataset(str(manifest))
    assert result["status"] == "completed"
    assert result["metrics_summary"]["attempted"] == 2
    assert result["metrics_summary"]["evaluated_count"] == 1
    assert result["metrics_summary"]["mean_per_image_chair_s"] == 0.0
    assert result["detailed_results"][1]["evaluation"]["chair_s"] is None


def test_dataset_manifest_reports_bad_records(tmp_path):
    manifest = tmp_path / "bad.json"
    manifest.write_text(json.dumps([{"image_path": "missing.png", "ground_truth_objects": "dog"}]))
    validation = DatasetEvaluator(FakeEvaluator()).validate_manifest(str(manifest))
    assert not validation["valid"]
    assert validation["valid_records"] == 0
    assert len(validation["errors"]) == 2


def test_database_migrates_legacy_inference_table(tmp_path, monkeypatch):
    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE inference_logs (id INTEGER PRIMARY KEY, timestamp TEXT, caption TEXT, "
            "word_count INTEGER, latency_ms REAL, device TEXT, chair_s_hallucination BOOLEAN, chair_i_score REAL)"
        )
    monkeypatch.setattr(database, "DB_NAME", str(db_path))

    database.init_db()
    database.log_inference(caption="A caption", evaluation_status="not_evaluated")
    with sqlite3.connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(inference_logs)")}
    assert {"image_id", "detected_objects", "evaluation_status", "chair_s", "metrics_json"} <= columns
    assert database.get_logs(1)[0]["caption"] == "A caption"


def test_caption_metrics_report_when_optional_metric_dependencies_are_missing():
    evaluator = CaptionMetricsEvaluator()
    if not evaluator.nltk_available or not evaluator.rouge_available:
        result = evaluator.evaluate_generation("a cat", "a cat")
        assert result["status"] == "partial"
        if not evaluator.nltk_available:
            assert result["metrics"]["BLEU-4"] == "unavailable_dependency"
        if not evaluator.rouge_available:
            assert result["metrics"]["ROUGE-L"] == "unavailable_dependency"
