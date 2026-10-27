# Installation and local run guide

## Requirements

- Python 3.11 is the version used by the Dockerfile. Python 3.10 or later is required by the type syntax in the source.
- Install dependencies from `requirements.txt`.
- The first run needs network access to retrieve the Hugging Face BLIP checkpoint and TorchVision Faster R-CNN weights unless the relevant caches already exist.
- Model initialization and inference require substantial memory. The API pipeline defaults to CPU; it does not automatically select CUDA or Apple Metal.

## Local setup

From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

On Windows, activate the environment with `.venv\Scripts\activate`. The API starts at `http://localhost:8000`; interactive API documentation is at `http://localhost:8000/docs` after startup succeeds.

Open `index.html` in a browser. If the browser restricts requests from a `file://` page, serve the repository directory with a local static server, for example `python -m http.server 8080`, and open `http://localhost:8080`. The page’s API base is configured by the `captionlens-api-base` meta tag.

## Docker

```bash
docker compose up --build
```

This starts the API only. It stores the SQLite file in the Compose named volume `captionlens-data`. Open the frontend separately as described above. The container needs network access to fetch model weights on its first start. No GPU runtime is configured in the Compose file.

## Readiness and common failures

- A model download or memory failure prevents application startup; inspect the API/container logs.
- The page displays an API status. If it reports the API unavailable, start the backend and reload or retry.
- `/health` reports the API’s device after model initialization.
- Uploads must be JPEG, PNG, or WebP and no larger than 10 MiB.
- The page is not served by the Docker image; port 8000 is the API port.

See [API reference](api-reference.md) and [architecture](architecture.md).
