import sqlite3
import json
from datetime import datetime

DB_NAME = "captionlens.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS inference_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            image_id TEXT,
            caption TEXT NOT NULL,
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
    ''')
    conn.commit()
    conn.close()

def log_inference(
    image_id=None,
    caption="",
    model_checkpoint="Salesforce/blip-image-captioning-base",
    latency_ms=0.0,
    device="cpu",
    detected_objects=None,
    evaluation_status="not_evaluated",
    evaluation_method=None,
    gt_source=None,
    chair_s=None,
    chair_i=None,
    metrics=None,
    error_msg=None
):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO inference_logs (
            timestamp, image_id, caption, model_checkpoint, latency_ms, device,
            detected_objects, evaluation_status, evaluation_method, gt_source,
            chair_s, chair_i, metrics_json, error_msg
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.utcnow.isoformat(),
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
        error_msg
    ))
    conn.commit()
    conn.close()

def get_logs(limit=50):
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM inference_logs ORDER BY id DESC LIMIT ?', (limit,))
    rows = cursor.fetchall()
    logs = [dict(row) for row in rows]
    conn.close()
    return logs