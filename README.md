# CaptionLens

CaptionLens is an image-captioning research prototype. Its local API uses pretrained BLIP to generate a caption and pretrained TorchVision Faster R-CNN to detect objects. A static HTML dashboard provides the upload interface and a literature overview. SQLite stores inference logs.

The repository also contains Python utilities for reference-based caption metrics, CHAIR-inspired object checks, dataset-manifest evaluation, and BLIP LoRA fine-tuning. These utilities are not all exposed through the web API. No CaptionLens benchmark results are claimed; the [research completion report](docs/version-1/audit/CaptionLens_Completion_Report.txt) says no CaptionLens experiments were run.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features and Implementation Status](#features-and-implementation-status)
3. [System Architecture](#system-architecture)
4. [Technology Stack](#technology-stack)
5. [Repository Structure](#repository-structure)
6. [Prerequisites and Installation](#prerequisites-and-installation)
7. [Using CaptionLens](#using-captionlens)
8. [API Documentation](#api-documentation)
9. [Research and Evaluation Methodology](#research-and-evaluation-methodology)
10. [Datasets](#datasets)
11. [Cloud-First Training and Resource Management](#cloud-first-training-and-resource-management)
12. [Fine-Tuning Pipeline](#fine-tuning-pipeline)
13. [Testing and Reproducibility](#testing-and-reproducibility)
14. [Limitations and Responsible Use](#limitations-and-responsible-use)
15. [Roadmap](#roadmap)
16. [References and Acknowledgements](#references-and-acknowledgements)
17. [License and Contributions](#license-and-contributions)

## Project Overview

The live workflow accepts one image, generates a caption, detects objects independently, returns the results, and records the inference metadata in SQLite. The detector’s boxes are not aligned to caption phrases; this project does not currently implement validated phrase grounding.

The web interface also contains static literature and dataset summaries. Those summaries are background material, not outputs from the local models. See the [architecture guide](docs/architecture.md) for the file-level flow.

## Features and Implementation Status

| Feature | Status |
|---|---|
| Upload one JPEG, PNG, or WebP image and preview it | Implemented in `index.html`; limited to 10 MiB |
| Generate a caption | Implemented by BLIP in `model_pipeline.py`, exposed at `POST /caption` |
| Detect objects and display boxes | Implemented by pretrained Faster R-CNN; independent of caption text |
| Record and retrieve inference logs | Implemented with SQLite through `database.py` and `GET /logs` |
| BLEU-4 and ROUGE-L | Implemented for one reference/candidate pair in `caption_metrics.py`; not exposed through the API |
| CHAIR-inspired object comparison | Implemented for one caption with supplied object labels in `evaluate.py`; not canonical CHAIR |
| Dataset-level captioning and object checks | Implemented in `dataset_evaluation.py` for JSON/JSONL image manifests; requires model inference |
| LoRA fine-tuning | Implemented as a manifest-driven script; adapter loading is not wired into the API |
| Historical model comparison in the dashboard | Static literature summaries; these models are not all runnable in CaptionLens |

The [documentation index](docs/README.md) links to architecture, installation, API, evaluation, dataset, cloud-resource, fine-tuning, and test guides.

## System Architecture

```text
Browser: index.html
  ├─ previews a selected image
  └─ POST /caption (multipart field: file)
       └─ app.py validates upload and calls model_pipeline.py
            ├─ BLIP → generated caption
            └─ Faster R-CNN → independent object boxes
       ├─ database.py → SQLite inference_logs
       └─ JSON response → browser updates text, timing, and boxes
```

The FastAPI lifespan initializes/migrates the database and loads the models before serving. The API runs on port 8000. The HTML page is separate; Docker Compose runs the API but does not serve the frontend. Full details are in [docs/architecture.md](docs/architecture.md).

## Technology Stack

- Python, FastAPI, Uvicorn, Pydantic
- PyTorch and TorchVision Faster R-CNN
- Hugging Face Transformers and pretrained `Salesforce/blip-image-captioning-base`
- Pillow for image decoding
- PEFT LoRA for optional fine-tuning
- NLTK and `rouge-score` for optional reference metrics
- SQLite for local inference logs
- HTML, JavaScript, Tailwind CDN, and Google Fonts CDN for the browser page
- Docker and Docker Compose for API packaging

No database server, user authentication, cloud job runner, or remote object storage is configured.

## Repository Structure

```text
.
├── app.py                    FastAPI application and HTTP routes
├── model_pipeline.py         BLIP captioning and Faster R-CNN detection
├── database.py               SQLite schema, migration, write, and read functions
├── caption_metrics.py        BLEU-4 and ROUGE-L utility
├── evaluate.py               CHAIR-inspired object-name evaluation
├── dataset_evaluation.py     JSON/JSONL manifest evaluation
├── fine_tune_lora.py         BLIP LoRA training CLI and dataset class
├── index.html                Browser interface and research dashboard
├── tests/test_api.py         API and utility tests using a fake inference pipeline
├── docs/                     Current guides and historical research snapshot
├── requirements.txt          Runtime and research dependencies
├── requirements-dev.txt      Test dependencies
├── Dockerfile                API container definition
└── docker-compose.yml        API service and persistent SQLite volume
```

Model weights, local databases, and virtual environments should remain local; `.gitignore` excludes those artifact paths. The historical review files are indexed in [docs/README.md](docs/README.md).

## Prerequisites and Installation

Python 3.11 is used by the Dockerfile. Python 3.10 or later is required by the source syntax. First startup needs network access to download pretrained model weights unless they are already cached.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Open `index.html` in a browser. If the browser blocks requests from a `file://` page, serve the repository directory with `python -m http.server 8080` and open `http://localhost:8080`. The API base URL is set by the `captionlens-api-base` meta tag in the HTML.

For Docker, run `docker compose up --build`. This starts only the API; the frontend still needs to be opened or served separately. The named volume `captionlens-data` stores the SQLite database across container replacement. See the [installation guide](docs/installation.md).

## Using CaptionLens

1. Start the API and wait for both pretrained models to load.
2. Open the HTML page and visit **Live Studio**.
3. Select a JPEG, PNG, or WebP image up to 10 MiB.
4. Choose **Run Caption + Detection**. The UI prevents duplicate requests while one is in progress and reports API errors.
5. Inspect the caption and detector boxes. These boxes show object detection; they are not verified caption phrase grounding.
6. Use **Clear image** to remove the selected file and reset the displayed result.

The other dashboard controls navigate the static research sections, filter the literature table, compare qualitative dataset profiles, select architecture explanations, or move through the in-page presenter slides. They do not run historical models or benchmark experiments.

## API Documentation

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Report API status and loaded device |
| `POST` | `/caption` | Accept multipart `file`, generate caption and detector boxes, and log the run |
| `GET` | `/logs?limit=10` | Return recent SQLite inference logs; limit range is 1–100 |

`POST /caption` returns `status`, `caption`, `word_count`, `latency_ms`, `device`, and `grounding_boxes`. Error responses include HTTP 415 for unsupported declared media type, 413 for files over 10 MiB, 400 for invalid image content, 422 for a missing file, 503 if the pipeline is unavailable, and 500 for inference/logging errors. Interactive docs are served by FastAPI at `/docs` while the API is running. See [docs/api-reference.md](docs/api-reference.md).

## Research and Evaluation Methodology

- `caption_metrics.py` calculates single-reference BLEU-4 and ROUGE-L when their dependencies are available. CIDEr, METEOR, and CLIPScore appear only as availability notices; they are not calculated by this module.
- `evaluate.py` extracts a limited COCO object vocabulary from a caption and compares it with caller-supplied object labels. It explicitly labels the result CHAIR-inspired. This should not be reported as canonical CHAIR without protocol validation.
- `dataset_evaluation.py` generates a caption for each valid manifest image, then runs the object-list check where labels are available. It reports per-image means, not a canonical corpus-level CHAIR score.
- The live API does not call these evaluators. A normal inference log is marked `not_evaluated`.

Metric inputs, outputs, and caveats are documented in [docs/research-methodology.md](docs/research-methodology.md).

## Datasets

No benchmark dataset or annotation set is bundled or downloaded automatically. The detector uses pretrained COCO category labels; that is not the same as evaluating on the COCO dataset. Dataset names in the literature dashboard are informational only.

`DatasetEvaluator` accepts user-supplied JSON or JSONL manifests with `image_path` and optional `ground_truth_objects` (a list of strings). Paths may be relative to the manifest. It does not use `caption` or `reference_captions` fields to calculate text metrics. See [docs/dataset-evaluation.md](docs/dataset-evaluation.md).

The historical [claim-verification log](docs/version-1/audit/CaptionLens_Claim_Verification.csv) and [literature table](docs/version-1/literature/CaptionLens_Literature_Yearwise_2021_2026.csv) separate prior-work claims from CaptionLens implementation results.

## Cloud-First Training and Resource Management

No cloud-provider integration or provisioning script is included. The API defaults to CPU. The LoRA script selects CUDA when PyTorch reports it available; otherwise it uses CPU. It does not configure MPS, mixed precision, gradient accumulation, or cloud storage.

Cloud execution is an optional way to supply compute, not an implemented service feature. Consider model memory, dataset storage, checkpoint retention, and data access controls before running a real training job. The [cloud compute guide](docs/cloud-training.md) describes what the code does and does not manage.

## Fine-Tuning Pipeline

`fine_tune_lora.py` loads BLIP base weights and applies LoRA to `query` and `value` modules. It requires a JSON or JSONL manifest of image paths and captions; an empty manifest is rejected rather than replaced with synthetic data.

```bash
python fine_tune_lora.py data/train.json \
  --epochs 2 --batch-size 2 --learning-rate 0.00005 \
  --output-dir ./fine_tuned_blip_lora
```

The saved adapter is not loaded by the live API. No training run or improvement claim is made by this README. See [docs/fine-tuning.md](docs/fine-tuning.md).

## Testing and Reproducibility

Install test tools and run the suite:

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q
```

The tests use a fake inference pipeline and do not validate model quality or detector accuracy. Record checkpoint IDs, dependency versions, device, dataset provenance, splits, and generation settings for research runs. Details are in [docs/testing.md](docs/testing.md).

## Limitations and Responsible Use

- Captions can omit details or hallucinate objects.
- Faster R-CNN’s boxes are independent of caption text; no region grounding accuracy is calculated.
- CHAIR-inspired scores depend on complete, consistently named object labels and are not validated as canonical CHAIR.
- BLEU-4 and ROUGE-L measure overlap with one supplied reference; they do not prove visual correctness.
- Uploads are processed in memory, while captions and detection metadata are persisted in SQLite. The `/logs` endpoint has no authentication; keep the service on a trusted network.
- Model downloads need network access on first load. CPU inference may be slow and memory-intensive.
- The repository’s literature materials contain pending/unverified bibliographic fields; check the claim-verification log before citing.

## Roadmap

- Validate the CHAIR-inspired implementation against a documented canonical protocol and annotations.
- Add a reproducible benchmark command that records model/configuration and dataset provenance.
- Add adapter loading and compare base vs. fine-tuned checkpoints on a held-out set.
- Add real phrase-grounding or region-level evaluation only with suitable annotations and validation.
- Pin dependencies and record experiment outputs without committing private data or large model weights.

These are proposed next steps, not implemented features.

## References and Acknowledgements

The repository contains a scoped literature review, year-wise paper table, claim-verification log, search log, and presentation under [`docs/version-1/`](docs/version-1/README.md). The review is not described as an exhaustive systematic review, and the completion report states that no CaptionLens experiments were run.

The implementation uses the Hugging Face Transformers BLIP checkpoint and TorchVision’s pretrained Faster R-CNN weights. Consult their respective model and library documentation for upstream licenses and usage terms before redistribution.

## License and Contributions

No project license is currently specified. Add a `LICENSE` file before granting reuse rights. Contributions should include focused tests and documentation that reflect behavior actually implemented in the code.
