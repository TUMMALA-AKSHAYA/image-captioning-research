# CaptionLens
### Evidence-Aware Image Captioning and Hallucination Evaluation

> *Every caption makes a claim. CaptionLens helps investigate the evidence.*

"Can we trust what an image captioning model says about an image?"

Modern vision-language models can generate fluent, syntactically complex descriptions of visual scenes. However, autoregressive text generation driven by statistical co-occurrence priors frequently produces **object hallucinations**—asserting the presence of objects, attributes, or spatial relationships that are entirely absent from the underlying visual pixels. 

**CaptionLens** is a modular research platform and evaluation toolkit designed to investigate factual reliability in image captioning. It combines conditional text generation (Salesforce BLIP-Base), object evidence extraction (TorchVision Faster R-CNN), COCO-normalized CHAIR object hallucination evaluation, standard n-gram metrics (BLEU-4, ROUGE-L), dataset manifest validation, FastAPI backend persistence, and an interactive research dashboard.

---

## Abstract and Research Motivation

Image captioning models ingest visual data and output natural language descriptions. While standard n-gram metrics (such as BLEU, CIDEr, and ROUGE) measure surface n-gram overlap against human references, they fail to explicitly penalize or explain factual errors where a model invents unsupported entities. 

The core research motivation behind CaptionLens stems from the **Generative Caption Enrichment (GCE) Dilemma**: while pre-training or fine-tuning on large synthetic corpora improves general multimodal reasoning and caption detail, it often inflates object hallucination rates. When visual features are ambiguous, autoregressive language decoders rely on strong statistical word co-occurrence priors (e.g., predicting *"table"* whenever *"coffee mug"* is present). Consequently, fine-tuning on grounded human-verified corpora (DOCCI, COCONut-PanCap) is required to enforce factual consistency.

CaptionLens investigates this problem by providing a transparent evaluation pipeline that decouples caption generation from evidence verification. By extracting object bounding boxes and confidences via an object detector and cross-checking generated caption tokens against trusted ground-truth object annotations using normalized CHAIR mechanics, CaptionLens helps researchers audit factual fidelity. 

*Note on Scope:* The current repository implements core inference pipelines, normalized CHAIR scoring, dataset manifest validators, reference-based metrics, FastAPI routes, and SQLite logging. Full-scale multi-GPU dataset benchmarking and LoRA fine-tuning convergence remain experimental or planned future work.

---

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
11. [Fine-Tuning Pipeline](#fine-tuning-pipeline)
12. [Testing and Reproducibility](#testing-and-reproducibility)
13. [Research Contributions and Project Status](#research-contributions-and-project-status)
14. [Limitations and Responsible Use](#limitations-and-responsible-use)
15. [Roadmap](#roadmap)
16. [References and Acknowledgements](#references-and-acknowledgements)
17. [License and Contributions](#license-and-contributions)

---

## Project Overview

CaptionLens is structured as an end-to-end workflow for analyzing image descriptions:
1. **Image Ingestion:** A user provides an image through the API or frontend interface.
2. **Caption Generation:** The Salesforce BLIP-Base model generates a conditional natural language caption.
3. **Evidence Detection:** A TorchVision Faster R-CNN detector identifies object candidates, bounding boxes, and confidence scores.
4. **Hallucination Evaluation:** The CHAIR evaluation module parses the generated caption, normalizes and extracts object tokens, and compares them against optional ground-truth annotations.
5. **Persistence & Logging:** Experiment runs, latencies, detected objects, and evaluation statuses are recorded in an SQLite database (`captionlens.db`).
6. **Interface:** The browser-based dashboard (`index.html`) visualizes the outputs, bounding boxes, and experiment logs.

### Scientific Distinction & Detector Limitations
An object detector's predictions do **not** constitute definitive ground truth. Faster R-CNN is constrained by its 80-class COCO vocabulary, can produce false positives or missed detections, and cannot verify subtle attributes, actions, or relational claims (e.g., color, count, spatial layout) solely through bounding boxes. CaptionLens treats object detections as auxiliary visual evidence rather than a complete verification oracle.

---

## Features and Implementation Status

| Feature | Implementation File | What It Does | Status |
| :--- | :--- | :--- | :--- |
| **BLIP Caption Generation** | `model_pipeline.py` | Loads Salesforce BLIP-Base for conditional text generation with deterministic vs. exploratory decoding. | Implemented & Tested |
| **Object Evidence Detection** | `model_pipeline.py` | Runs TorchVision Faster R-CNN (ResNet-50 FPN) to extract object bounding boxes and labels. | Implemented & Tested |
| **Normalized CHAIR Evaluator** | `evaluate.py` | Computes sentence-level ($CHAIR_s$) and instance-level ($CHAIR_i$) hallucination scores with synonym and compound handling. | Implemented & Tested |
| **BLEU-4 & ROUGE-L Metrics** | `caption_metrics.py` | Computes reference-based n-gram overlap scores via NLTK and rouge-score with dependency fallbacks. | Implemented & Tested |
| **Dataset Manifest Validation** | `dataset_evaluation.py` | Validates JSON/JSONL dataset splits (DOCCI and COCONut manifest schemas). | Implemented & Tested |
| **FastAPI Backend** | `app.py` | Exposes REST endpoints for health checks, caption generation, evaluation, and log retrieval. | Implemented & Tested |
| **SQLite Persistence** | `database.py` | Initializes schema and records structured inference and evaluation logs in `captionlens.db`. | Implemented & Tested |
| **Interactive Dashboard** | `index.html` | Single-page frontend featuring live testing studio, literature survey, dataset analytics, and presentation deck. | Implemented & Tested |
| **LoRA Fine-Tuning** | `fine_tune_lora.py` | PEFT LoRA adapter training configuration script for vision-language models. | Experimental / Scaffolded |
| **Automated Tests** | `tests/test_api.py` | Pytest suite validating API endpoints, CHAIR logic, and database logging. | Implemented & Tested |
| **Docker Deployment** | `Dockerfile`, `docker-compose.yml` | Container build definitions and service orchestration configuration. | Implemented |

---

## System Architecture

The following Mermaid diagram outlines the verified data flow for interactive inference, evaluation, and logging:

```mermaid
graph TD
    A[Client Request / Image Upload] --> B[FastAPI Backend (/caption)]
    B --> C[Model Pipeline: model_pipeline.py]
    C -->|Image Tensors| D[Salesforce BLIP-Base Captioner]
    C -->|Image Tensors| E[TorchVision Faster R-CNN Detector]
    D --> F[Generated Caption Text]
    E --> G[Bounding Boxes & Labels]
    F & G --> H[Evaluation Engine: evaluate.py]
    H --> I[SQLite Experiment Logger: database.py]
    I --> J[JSON API Response]
    J --> K[Frontend Dashboard: index.html]
Note: Offline dataset-level batch evaluations (dataset_evaluation.py) and LoRA training (fine_tune_lora.py) operate as standalone modules rather than running during standard live API requests.

---

## Technology Stack

Programming Language: Python 3.10+

Deep Learning Framework: PyTorch, TorchVision

Vision-Language Model: Hugging Face Transformers (BlipProcessor, BlipForConditionalGeneration)

Object Detection: TorchVision Faster R-CNN (ResNet-50 FPN weights)

Backend & API: FastAPI, Uvicorn, Pydantic

Evaluation & NLP: NLTK (BLEU), rouge-score (ROUGE-L), custom regex parsing

Database & Persistence: SQLite (sqlite3)

Frontend: Tailwind CSS, HTML5 Canvas, Vanilla JavaScript

Testing & Deployment: Pytest, FastAPI TestClient, Docker, Docker Compose
