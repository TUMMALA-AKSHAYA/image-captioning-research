import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any

DB_NAME = os.environ.get("CAPTIONLENS_DB_PATH", "captionlens.db")


def init_db() -> None:
    """Create the inference log table and add columns from newer schema versions."""
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS inference_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                image_id TEXT,
                caption TEXT,
                model_checkpoint TEXT,
                latency_ms REAL,
                device TEXT,
                detected_objects TEXT,
                evaluation_status TEXT,
                evaluation_method TEXT,
                gt_source TEXT,
                chair_s REAL,
                chair_i REAL,
                metrics_json TEXT,
                error_msg TEXT
            )
            """
        )
        existing_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(inference_logs)")
        }
        migrations = {
            "image_id": "TEXT",
            "model_checkpoint": "TEXT",
            "detected_objects": "TEXT",
            "evaluation_status": "TEXT",
            "evaluation_method": "TEXT",
            "gt_source": "TEXT",
            "chair_s": "REAL",
            "chair_i": "REAL",
            "metrics_json": "TEXT",
            "error_msg": "TEXT",
        }
        for name, column_type in migrations.items():
            if name not in existing_columns:
                conn.execute(f"ALTER TABLE inference_logs ADD COLUMN {name} {column_type}")


def log_inference(
    image_id: str | None = None,
    caption: str = "",
    model_checkpoint: str = "Salesforce/blip-image-captioning-base",
    latency_ms: float = 0.0,
    device: str = "cpu",
    detected_objects: list[str] | None = None,
    evaluation_status: str = "not_evaluated",
    evaluation_method: str | None = None,
    gt_source: str | None = None,
    chair_s: float | None = None,
    chair_i: float | None = None,
    metrics: dict[str, Any] | None = None,
    error_msg: str | None = None,
) -> None:
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute(
            """
            INSERT INTO inference_logs (
                timestamp, image_id, caption, model_checkpoint, latency_ms, device,
                detected_objects, evaluation_status, evaluation_method, gt_source,
                chair_s, chair_i, metrics_json, error_msg
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                image_id,
                caption,
                model_checkpoint,
                latency_ms,
                device,
                json.dumps(detected_objects or []),
                evaluation_status,
                evaluation_method,
                gt_source,
                chair_s,
                chair_i,
                json.dumps(metrics or {}),
                error_msg,
            ),
        )


def get_logs(limit: int = 50) -> list[dict[str, Any]]:
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    with sqlite3.connect(DB_NAME) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM inference_logs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(row) for row in rows]
