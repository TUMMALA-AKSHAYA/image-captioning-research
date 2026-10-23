import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from model_pipeline import ProductionCaptionPipeline
from evaluate import ProductionEvaluator
from database import init_db, log_inference, get_recent_logs

app = FastAPI(
    title="CaptionLens Production API",
    description="Vision-Language Inference & SQLite Logging Microservice",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Initializing PyTorch Model Pipeline & SQLite Database...")
init_db()
pipeline = ProductionCaptionPipeline()
evaluator = ProductionEvaluator()

class GroundingBox(BaseModel):
    label: str
    confidence: float
    x: float
    y: float
    w: float
    h: float
    color: str

class CaptionResponse(BaseModel):
    status: str
    caption: str
    word_count: int
    latency_ms: float
    device: str
    grounding_boxes: List[GroundingBox]

@app.get("/health")
def health_check():
    return {"status": "online", "device": str(pipeline.device)}

@app.post("/caption", response_model=CaptionResponse)
async def generate_caption_endpoint(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = pipeline.generate_caption(image_bytes=image_bytes)
    
    # Log inference run to SQLite
    log_inference(result)
    
    return CaptionResponse(status="success", **result)

@app.get("/logs")
def get_logs_endpoint(limit: int = 10):
    """Retrieves past experiment logs from SQLite."""
    return {"status": "success", "logs": get_recent_logs(limit)}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)