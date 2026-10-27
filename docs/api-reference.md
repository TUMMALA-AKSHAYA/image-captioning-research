# API reference

The API is defined in `app.py` and runs on port 8000. There is no authentication. CORS permits all origins without credentials, for local frontend use.

## `GET /health`

Returns the loaded pipeline device:

```json
{"status":"online","device":"cpu"}
```

If the pipeline is unavailable, the route returns HTTP 503 with a `detail` message. During normal startup, model loading occurs in the application lifespan before requests are served.

## `POST /caption`

Send `multipart/form-data` with one required file field named `file`.

- Supported declared media types: `image/jpeg`, `image/png`, `image/webp`.
- Maximum upload size: 10 MiB.
- The server also verifies that Pillow can decode the contents.

Example with curl:

```bash
curl -X POST http://localhost:8000/caption \
  -F 'file=@/path/to/image.jpg;type=image/jpeg'
```

Success response fields:

```json
{
  "status": "success",
  "caption": "...",
  "word_count": 12,
  "latency_ms": 123.45,
  "device": "cpu",
  "grounding_boxes": [
    {"label":"dog","confidence":0.91,"x":25.0,"y":30.0,"w":120.0,"h":140.0,"color":"#10b981"}
  ]
}
```

The values above illustrate the response shape; they are not measured project results. Boxes use the frontend canvas’s 400×300 coordinate space. They are detector output, not caption phrase alignment. The API logs the caption and detected labels with `evaluation_status="not_evaluated"`; it does not calculate a CHAIR score.

Errors: HTTP 415 for unsupported declared media types, HTTP 413 for uploads over 10 MiB, HTTP 400 for undecodable image bytes, HTTP 422 for a missing required file, HTTP 503 if the pipeline is unavailable, and HTTP 500 for inference or logging failures. Internal model exceptions are logged server-side; the response does not include their raw text.

## `GET /logs?limit=10`

Returns recent rows from `inference_logs`. `limit` must be from 1 through 100. The response is `{ "status": "success", "logs": [...] }`. Rows can contain captions and other inference metadata; keep this local service on a trusted network.

## Not exposed as HTTP routes

`CaptionMetricsEvaluator` and `DatasetEvaluator` are Python utilities. There is no HTTP endpoint for reference metrics or dataset evaluation in this repository.
