# CaptionLens — Evidence-Aware Image Captioning & Hallucination Evaluation

**CaptionLens** is an advanced image-captioning research and evaluation platform designed to generate descriptive text, detect visual evidence, and rigorously quantify object hallucinations using normalized COCO CHAIR metrics.

---

## 🚀 Key Features
* **Vision-Language Baseline:** Integrates Salesforce BLIP-Base for conditional caption generation with deterministic vs. exploratory decoding controls.
* **Object Evidence Detection:** Utilizes TorchVision Faster R-CNN (ResNet-50 FPN) to extract bounding boxes, confidence scores, and object labels.
* **Normalized CHAIR Evaluator:** Implements official COCO CHAIR object hallucination evaluation with synonym mapping, singularization, compound word handling (`hot dog`), and explicit missing-value semantics (`not_evaluated`).
* **Dataset Manifest Validation:** Validates JSON/JSONL dataset splits for DOCCI and COCONut-PanCap formats.
* **Standard Caption Metrics:** Modular computation of BLEU-4 and ROUGE-L with graceful dependency fallbacks.
* **SQLite Persistence:** Logs inference runs, latencies, evaluation statuses, and computed metrics for complete reproducibility.
* **Interactive Dashboard (`index.html`):** Fully functional web UI featuring a live testing studio, 5-year literature survey, dataset analytics, and executive presentation deck[cite: 27, 28].

---

## 🛠️ Tech Stack
* **Backend:** Python, FastAPI, PyTorch, TorchVision, Hugging Face Transformers
* **Evaluation & Testing:** NumPy, NLTK, Pytest
* **Database:** SQLite (`captionlens.db`)
* **Frontend:** Tailwind CSS, HTML5 Canvas

---

## ⚙️ Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/image-captioning-research.git](https://github.com/YOUR_USERNAME/image-captioning-research.git)
   cd image-captioning-research
