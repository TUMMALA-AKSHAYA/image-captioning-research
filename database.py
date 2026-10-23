import sqlite3
from datetime import datetime
from typing import Dict, Any, List

DB_NAME = "captionlens.db"

def init_db():
    """Initializes the SQLite database and creates the inference logs table."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS inference_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            caption TEXT,
            word_count INTEGER,
            latency_ms REAL,
            device TEXT,
            chair_s_hallucination BOOLEAN,
            chair_i_score REAL
        )
    """)
    conn.commit()
    conn.close()

def log_inference(result: Dict[str, Any], evaluation: Dict[str, Any] = None):
    """Logs a completed captioning or evaluation run into SQLite."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    timestamp = datetime.now().isoformat()
    caption = result.get("caption", "")
    word_count = result.get("word_count", 0)
    latency_ms = result.get("latency_ms", 0.0)
    device = result.get("device", "cpu")
    
    chair_s = evaluation.get("chair_s_has_hallucination", False) if evaluation else False
    chair_i = evaluation.get("chair_i_instance_score_pct", 0.0) if evaluation else 0.0

    cursor.execute("""
        INSERT INTO inference_logs (timestamp, caption, word_count, latency_ms, device, chair_s_hallucination, chair_i_score)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (timestamp, caption, word_count, latency_ms, device, chair_s, chair_i))
    
    conn.commit()
    conn.close()

def get_recent_logs(limit: int = 10) -> List[Dict[str, Any]]:
    """Retrieves recent inference history from SQLite."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM inference_logs ORDER BY id DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

if __name__ == "__main__":
    init_db()
    print("Database module initialized successfully.")