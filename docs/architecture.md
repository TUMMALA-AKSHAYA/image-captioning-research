# Architecture and data flow

CaptionLens is a static HTML page connected to a local FastAPI service. The API uses pretrained BLIP for caption generation and pretrained TorchVision Faster R-CNN for object detection. It records inference metadata in SQLite.

```text
Browser (index.html)
  ├─ selects and previews an image
  └─ POST /caption (multipart field: file)
       └─ app.py validates and reads the upload
            └─ model_pipeline.py
                 ├─ BLIP processor → BLIP caption generation
                 └─ Faster R-CNN → independent object boxes
            ├─ database.py → SQLite inference_logs
            └─ JSON response → browser caption, boxes, timing
```

## Startup

`app.py` creates the FastAPI object at import time. Its lifespan handler initializes/migrates SQLite and loads `ProductionCaptionPipeline`. The API is not ready to serve until both pretrained model families have loaded. `python app.py` starts Uvicorn on port 8000 with reload enabled for development. The Docker command runs Uvicorn without reload.

The browser is separate from the API container. `index.html` defaults to `http://localhost:8000`; that address can be changed in its `captionlens-api-base` meta tag. The Docker image does not host the HTML page.

## Request processing

`POST /caption` accepts JPEG, PNG, and WebP uploads up to 10 MiB. Pillow verifies that the bytes are a decodable image. The pipeline converts the image to RGB, generates a caption, then runs Faster R-CNN over the same image. Detection boxes are returned in a 400×300 coordinate space for the canvas.

The detector output is **not phrase grounding**: no caption word or phrase is matched to a box, and no mask or box ground truth is used to score localization. The endpoint does not run BLEU, ROUGE, or CHAIR evaluation. Its database record is explicitly marked `not_evaluated`.

## Persistence

`database.py` uses `CAPTIONLENS_DB_PATH`, defaulting to `captionlens.db` relative to the process working directory. Docker Compose mounts a named volume at `/data` and sets the database path to `/data/captionlens.db`. Uploaded image bytes are processed in memory and are not saved by the API. Captions, detector labels, timing, and status are saved in the SQLite log.

## Related guides

- [API request and response details](api-reference.md)
- [Metric limitations](research-methodology.md)
- [Local and Docker setup](installation.md)
