# Dataset manifest and evaluation

`DatasetEvaluator` in `dataset_evaluation.py` supports JSON and JSONL manifests. It generates a new caption for each listed image with the BLIP model and, when object labels are supplied, runs the CHAIR-inspired object-list comparison.

## Manifest format

JSON is a list of records:

```json
[
  {"image_path":"images/example.jpg","ground_truth_objects":["person","bicycle"]},
  {"image_path":"images/another.png"}
]
```

JSONL uses one JSON object per non-empty line with the same fields. `image_path` is required and must name an existing image file. Relative paths are resolved from the manifest’s directory. `ground_truth_objects` is optional and, when present, must be a list of strings. Other fields are ignored.

No COCO, Flickr30K, NoCaps, DOCCI, MIMIC-CXR, or other dataset is bundled or loaded automatically. The manifest is the caller’s data; the evaluator does not download a dataset or infer its annotation source.

## Calling it

```python
from dataset_evaluation import DatasetEvaluator

evaluator = DatasetEvaluator()
print(evaluator.validate_manifest("data/manifest.jsonl"))
results = evaluator.evaluate_dataset("data/manifest.jsonl")
```

`evaluate_dataset()` performs model inference, so it may take substantial time and memory. `validate_manifest()` checks file existence and annotation shape but does not decode every image. Invalid records are reported and valid records can still be evaluated. Each detailed result contains the generated caption and either an evaluation record or an error.

`attempted` counts valid manifest records sent through evaluation. `evaluated_count` counts successful captions with object annotations. Records without annotations are captioned but have null metric values and are excluded from the mean. The returned CHAIR-like summary is the arithmetic mean of per-image values, not the canonical corpus-level CHAIR protocol.

The utility does not compute BLEU/ROUGE against reference captions. Those functions are separate in [`research-methodology.md`](research-methodology.md). No HTTP endpoint or command-line entry point invokes dataset evaluation.
