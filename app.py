import logging
import uuid
import warnings
from contextlib import asynccontextmanager
from io import BytesIO

import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from database import get_logs, init_db, log_inference
from model_pipeline import ProductionCaptionPipeline

logger = logging.getLogger("captionlens.api")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


@asynccontextmanager
async def lifespan(application: FastAPI):
    init_db()
    application.state.pipeline = ProductionCaptionPipeline()
    yield
    application.state.pipeline = None


app = FastAPI(
    title="CaptionLens API",
    description="Local image captioning and object detection API.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)


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
    grounding_boxes: list[GroundingBox]


@app.get("/health")
def health_check(request: Request):
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Caption model is starting up.")
    return {"status": "online", "device": str(pipeline.device)}


@app.post("/caption", response_model=CaptionResponse)
async def generate_caption_endpoint(request: Request, file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Unsupported image type. Upload a JPEG, PNG, or WebP image."
        )

    try:
        image_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await file.close()

    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MB upload limit.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(image_bytes)) as image:
                image.load()
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ):
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")

    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Caption model is starting up.")

    try:
        result = await run_in_threadpool(pipeline.generate_caption, image_bytes=image_bytes)
    except Exception:
        logger.exception("Caption generation failed")
        raise HTTPException(status_code=500, detail="Caption generation failed. Check the API logs.")

    boxes = result.get("grounding_boxes", [])
    try:
        log_inference(
            image_id=str(uuid.uuid4()),
            caption=result["caption"],
            model_checkpoint="Salesforce/blip-image-captioning-base",
            latency_ms=result["latency_ms"],
            device=result["device"],
            detected_objects=[box["label"] for box in boxes],
            evaluation_status="not_evaluated",
            evaluation_method=None,
            gt_source=None,
            chair_s=None,
            chair_i=None,
            metrics={"word_count": result["word_count"]},
        )
    except Exception:
        logger.exception("Could not save inference log")
        raise HTTPException(status_code=500, detail="Caption was generated but could not be logged.")

    return CaptionResponse(
        status="success",
        caption=result["caption"],
        word_count=result["word_count"],
        latency_ms=result["latency_ms"],
        device=result["device"],
        grounding_boxes=boxes,
    )


@app.get("/logs")
def get_logs_endpoint(limit: int = Query(10, ge=1, le=100)):
    return {"status": "success", "logs": get_logs(limit)}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
