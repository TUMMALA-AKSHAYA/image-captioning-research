# CaptionLens
### Evidence-Aware Image Captioning and Hallucination Evaluation

> *Every caption makes a claim. CaptionLens helps investigate the evidence.*

"Can we trust what an image captioning model says about an image?"

Modern vision-language models can generate fluent, syntactically complex descriptions of visual scenes. However, autoregressive text generation driven by statistical co-occurrence priors frequently produces **object hallucinations**—asserting the presence of objects, attributes, or spatial relationships that are entirely absent from the underlying visual pixels. 

**CaptionLens** is a modular research platform and evaluation toolkit designed to investigate factual reliability in image captioning. It combines conditional text generation (Salesforce BLIP-Base), object evidence extraction (TorchVision Faster R-CNN), COCO-normalized CHAIR object hallucination evaluation, standard n-gram metrics (BLEU-4, ROUGE-L), dataset manifest validation, FastAPI backend persistence, and an interactive research dashboard.

---

## Abstract and Research Motivation

Image captioning models ingest visual data and output natural language descriptions. While standard n-gram metrics (such as BLEU, CIDEr, and ROUGE) measure surface n-gram overlap against human references, they fail to explicitly penalize or explain factual errors where a model invents unsupported entities. 

The core research motivation behind CaptionLens stems from the **Generative Caption Enrichment (GCE) Dilemma**: while pre-training or fine-tuning on large synthetic corpora improves general multimodal reasoning and caption detail, it often inflates object hallucination rates. When visual features are ambiguous, autoregressive language decoders rely on strong statistical priors rather than direct visual grounding.

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
Note: Offline dataset-level batch evaluations (dataset_evaluation.py) and LoRA training (fine_tune_lora.py) operate as standalone modules rather than running during standard live API requests.Technology StackProgramming Language: Python 3.10+Deep Learning Framework: PyTorch, TorchVisionVision-Language Model: Hugging Face Transformers (BlipProcessor, BlipForConditionalGeneration)Object Detection: TorchVision Faster R-CNN (ResNet-50 FPN weights)Backend & API: FastAPI, Uvicorn, PydanticEvaluation & NLP: NLTK (BLEU), rouge-score (ROUGE-L), custom regex parsingDatabase & Persistence: SQLite (sqlite3)Frontend: Tailwind CSS, HTML5 Canvas, Vanilla JavaScriptTesting & Deployment: Pytest, FastAPI TestClient, Docker, Docker ComposeRepository StructurePlaintextimage-captioning-research/
├── app.py                 # FastAPI web application and endpoint routes
├── model_pipeline.py      # Production caption generation and Faster R-CNN evidence detector
├── evaluate.py            # Normalized COCO CHAIR hallucination evaluation engine
├── dataset_evaluation.py  # Dataset manifest validator and batch evaluation runner
├── caption_metrics.py     # BLEU-4 and ROUGE-L reference-based caption evaluators
├── fine_tune_lora.py      # PEFT LoRA fine-tuning training script scaffold
├── database.py            # SQLite database initialization and logging functions
├── index.html             # Interactive frontend research dashboard and studio
├── Dockerfile             # Container definition for the FastAPI application
├── docker-compose.yml     # Service orchestration configuration
├── requirements.txt       # Python package dependencies
└── tests/
    └── test_api.py        # Pytest suite for API endpoints and evaluation logic
Prerequisites and InstallationPrerequisitesPython 3.10 or higher.pip package manager.(Optional) NVIDIA GPU with CUDA support for accelerated inference. CPU execution is fully supported by default.Installation & Setup InstructionsClone the Repository:Bashgit clone [https://github.com/YOUR_USERNAME/image-captioning-research.git](https://github.com/YOUR_USERNAME/image-captioning-research.git)
cd image-captioning-research
(Replace YOUR_USERNAME with your actual GitHub handle).Create and Activate a Virtual Environment:macOS / Linux:Bashpython -m venv .venv
source .venv/bin/activate
Windows (Command Prompt / PowerShell):DOSpython -m venv .venv
.venv\Scripts\activate
Install Dependencies:Bashpip install --upgrade pip
pip install -r requirements.txt
Run Automated Tests:Bashpython -m pytest -v
Start the FastAPI Backend:Bashuvicorn app:app --reload --port 8000
Open the Frontend:
Open index.html directly in any modern web browser to access the research studio and interface with the local FastAPI server.Docker Setup (Optional):Bashdocker-compose up --build
Using CaptionLensLaunch the Interface: Open index.html in your browser.Select or Upload Visual Samples: Choose from built-in preset scenes (Urban Street, Kitchen Scene, Wildlife Park) or upload a custom image file.Select Model Era: Choose between architectural eras ranging from classical CNN-GRU baselines to modern grounded models.Inspect Evidence & Metrics: Review generated captions, hover over grounding tags to inspect spatial bounding box overlays on the canvas, and examine hallucination risk indicators.Audit History: Review persistent experiment logs stored in the SQLite backend.API DocumentationThe FastAPI backend exposes the following verified endpoints:1. Health CheckMethod & Path: GET /healthPurpose: Checks service status and active runtime device (cpu or cuda).Response Example (200 OK):JSON{
  "status": "online",
  "device": "cpu"
}
2. Caption Generation & Evidence DetectionMethod & Path: POST /captionPurpose: Accepts an uploaded image file (multipart/form-data), runs BLIP captioning and Faster R-CNN object detection, records the run in SQLite, and returns structured results.Request Format: Form-data with key file (image file).Response Example (200 OK):JSON{
  "caption": "A city street with a red bus and people on a sidewalk.",
  "word_count": 11,
  "latency_ms": 142.5,
  "grounding_boxes": [
    {"label": "person", "confidence": 0.95, "x": 30, "y": 120, "w": 100, "h": 140, "color": "#10b981"},
    {"label": "bus", "confidence": 0.98, "x": 140, "y": 80, "w": 180, "h": 160, "color": "#ef4444"}
  ]
}
3. Evaluate Caption HallucinationsMethod & Path: POST /evaluatePurpose: Evaluates object hallucination for a given caption against optional ground-truth object annotations.Request Body (application/json):JSON{
  "caption": "A player eating a hot dog next to a bicycle.",
  "ground_truth_objects": ["person", "hot dog", "bicycle"]
}
Response Example (200 OK):JSON{
  "evaluation_status": "evaluated",
  "evaluation_method": "CHAIR-inspired COCO normalized",
  "ground_truth_provided": true,
  "mentioned_objects": ["person", "hot dog", "bicycle"],
  "hallucinated_objects": [],
  "chair_s": 0.0,
  "chair_i": 0.0,
  "message": "Evaluation successfully completed against provided ground-truth objects."
}
4. Retrieve Experiment LogsMethod & Path: GET /logsPurpose: Queries SQLite persistence and returns recent inference and evaluation records.Query Parameters: limit (optional integer, default 50).Response Example (200 OK):JSON[
  {
    "id": 1,
    "timestamp": "2026-10-27T12:55:00.000000",
    "image_id": null,
    "caption": "A city street with a red bus...",
    "model_checkpoint": "Salesforce/blip-image-captioning-base",
    "latency_ms": 142.5,
    "device": "cpu",
    "evaluation_status": "not_evaluated",
    "chair_s": null,
    "chair_i": null
  }
]
Research and Evaluation MethodologyCaptionLens separates evaluation into complementary axes:A. CHAIR (Caption Hallucination Assessment with Image Recognition)Object hallucination is evaluated using a CHAIR-inspired protocol (evaluate.py). The module extracts noun tokens from generated captions, applies singularization, maps common synonyms (e.g., player $\to$ person), handles compound words (hot dog, sports ball), and compares them against trusted ground-truth object annotations.$CHAIR_s$ (Sentence-level): The proportion of evaluated captions that contain at least one hallucinated object ($1.0$ if hallucinated, else $0.0$).$CHAIR_i$ (Instance-level): The proportion of mentioned object instances that are unsupported by ground-truth annotations.Missing-Value Semantics: If ground-truth annotations are omitted, CaptionLens explicitly returns not_evaluated and null values rather than falsely recording a zero hallucination score.B. BLEU-4 and ROUGE-LStandard reference-based n-gram metrics (caption_metrics.py) measure surface textual similarity against human reference captions using NLTK and rouge-score, reporting graceful fallback messages if dependencies are missing.C. Visual Evidence & LimitationsObject detector bounding boxes provide localizing evidence but do not constitute comprehensive semantic verification.DatasetsThe repository provides dataset manifest validation scaffolding (dataset_evaluation.py) designed for JSON/JSONL dataset splits:MS COCO (2014): Standard benchmark containing human-annotated image captions.ShareGPT4V: Large-scale dataset with synthetic long-form descriptions.DOCCI (Description of Context-in-Context Images): Google-curated detailed contrastive image descriptions.COCONut-PanCap: Panoptic mask-grounded image captioning corpus.Status: The repository includes schema validation and batch evaluation utilities for manifest records, but does not automatically download multi-gigabyte raw datasets.Fine-Tuning PipelineThe script fine_tune_lora.py provides a Parameter-Efficient Fine-Tuning (PEFT) scaffold utilizing Hugging Face peft and transformers to adapt vision-language models via Low-Rank Adaptation (LoRA).Status: Configured as a structural training scaffold; full-scale GPU training convergence and benchmark evaluations on held-out test splits remain experimental.Testing and ReproducibilityExecuting Tests:Bashpython -m pytest -v
Test Coverage: The test suite (tests/test_api.py) verifies FastAPI health endpoints, CHAIR normalization rules (synonyms, compound words, missing-value semantics), dataset manifest validation, and SQLite log insertion.Reproducibility: Deterministic inference is controlled via explicit decoding parameters (do_sample=False, num_beams=3).Research Contributions and Project StatusImplemented & Verified: BLIP conditional captioning inference, Faster R-CNN object detection, normalized CHAIR evaluation, SQLite logging, FastAPI endpoints, pytest suite, and interactive frontend studio.Experimental / In Progress: Dataset-level batch evaluation pipelines and LoRA fine-tuning adapter validation.Future Work: Multi-GPU training convergence studies and cross-modal alignment (CLIPScore) integration.Limitations and Responsible UseDetector Vocabulary Constraints: Faster R-CNN is bounded by the 80 COCO object categories; novel, fine-grained, or occluded objects may trigger false hallucination flags.Metric Dependency: CHAIR metrics rely heavily on text tokenization, noun chunking, and synonym dictionary coverage.Hardware Requirements: Real-time object detection and neural text generation require adequate CPU resources or a compatible NVIDIA GPU.Factual Scope: Object-level metrics quantify entity presence/absence but do not capture complex physical interactions, temporal events, or subjective attributes.Roadmap[x] Core BLIP inference and Faster R-CNN evidence integration[x] COCO-normalized CHAIR hallucination evaluator with missing-value semantics[x] FastAPI REST endpoints and SQLite experiment persistence[x] Interactive editorial-tech research dashboard (index.html)[x] Pytest test suite for API routes and CHAIR normalization[ ] Large-scale dataset batch evaluation across DOCCI and COCONut splits[ ] PEFT LoRA fine-tuning convergence validationReferences and AcknowledgementsLi, J., Li, D., Xiong, C., & Hoi, S. (2022). BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation. Proceedings of the 39th International Conference on Machine Learning (ICML). arXiv:2201.12086Rohrbach, A., Hendricks, L. A., Burns, K., Darrell, T., & Saenko, K. (2018). Object Hallucination in Captioning. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). arXiv:1809.02156Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. Advances in Neural Information Processing Systems (NeurIPS). arXiv:1506.01497Chen, X., et al. (2024). DOCCI: Description of Context-in-Context Images. Google Research.License and ContributionsLicense: Not specified in retrieved sources / pending license selection. Please check the repository root for any newly added license files.Contributions: Research extensions, bug fixes, and evaluation improvements are welcome. Please ensure all modifications pass the existing pytest suite before opening pull requests.
