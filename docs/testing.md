# Testing and reproducibility

The tests use a fake caption pipeline for API requests, so they do not download model weights or perform real model inference. Evaluator unit tests exercise the deterministic object-name rules. Dataset tests use a fake evaluator and temporary image/manifest files.

Install the project dependencies and development test tools, then run:

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q
```

The checked-in `requirements.txt` contains minimum versions, not a complete lockfile. For reproducible research, record Python, PyTorch/TorchVision, Transformers, PEFT, metric package versions, checkpoint identifiers, device, manifest/annotation provenance, and generation settings with each run.

The automated tests do not verify generated captions, detector quality, benchmark scores, Docker image startup, or browser rendering. Those require explicit runtime checks. Starting the real API loads pretrained checkpoints; running `DatasetEvaluator.evaluate_dataset()` performs inference over every valid manifest image.
