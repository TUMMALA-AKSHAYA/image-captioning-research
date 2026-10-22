import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model_pipeline import ProductionCaptionPipeline

# Initialize FastAPI Application
app = FastAPI(
    title="Image Captioning Microservice API",
    description="Production REST endpoint for generating visual captions from uploaded images.",
    version="1.0.0"
)

# Enable Cross-Origin Resource Sharing (CORS) for front-end integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load PyTorch Pipeline Context on Startup
print("Initializing PyTorch Model Pipeline...")
pipeline = ProductionCaptionPipeline()
print(f"Model successfully loaded on device: {pipeline.device}")


class CaptionResponse(BaseModel):
    status: str
    caption: str
    word_count: int
    latency_ms: float
    device: str


@app.get("/health")
def health_check():
    return {
        "status": "online",
        "device": str(pipeline.device),
        "vocab_size": pipeline.vocab_size
    }


@app.post("/caption", response_model=CaptionResponse)
async def generate_caption_endpoint(
    file: UploadFile = File(...),
    temperature: float = Query(0.7, ge=0.1, le=2.0, description="Sampling temperature"),
    top_k: int = Query(5, ge=0, le=50, description="Top-k nucleus filtering parameter")
):
    # Validate MIME Type
    if file.content_type not in ["image/jpeg", "image/jpg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=400, 
            detail="Invalid file format. Only JPEG, PNG, and WebP images are supported."
        )

    try:
        image_bytes = await file.read()
        result = pipeline.generate_caption(
            image_bytes=image_bytes,
            temperature=temperature,
            top_k=top_k
        )
        return CaptionResponse(
            status="success",
            caption=result["caption"],
            word_count=result["word_count"],
            latency_ms=result["latency_ms"],
            device=result["device"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)