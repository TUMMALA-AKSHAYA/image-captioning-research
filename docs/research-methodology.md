# Research and evaluation methodology

## What the running API does

`ProductionCaptionPipeline.generate_caption()` in `model_pipeline.py` generates caption text using the pretrained checkpoint `Salesforce/blip-image-captioning-base`. It also runs pretrained TorchVision Faster R-CNN and returns object labels, confidence values, and boxes. These are two independent model outputs.

The detector boxes are not matched to caption phrases. The code has no segmentation model, phrase-grounding model, or box/mask localization evaluation. The frontend labels them as detected objects and displays the CHAIR status as “Not evaluated.”

## Reference-based metrics

`CaptionMetricsEvaluator.evaluate_generation()` in `caption_metrics.py` accepts one reference caption and one generated caption.

- **BLEU-4:** Uses NLTK `sentence_bleu` with the library’s default four-gram weights and smoothing method 1. It is a single-reference sentence score, not a dataset corpus score.
- **ROUGE-L:** Uses the F-measure from `rouge_score`’s `rougeL` scorer with stemming enabled.
- The code returns an availability marker for CIDEr, METEOR, and CLIPScore; it does not calculate those metrics.
- NLTK and `rouge-score` are listed in `requirements.txt`. Exceptions during a metric calculation are logged and represented as unavailable (`None` internally).

These metrics are reference-overlap measures; they do not establish that an object or relation is visually correct. The live `/caption` API does not call this evaluator.

## Object hallucination check

`ProductionEvaluator.normalize_and_extract_objects()` in `evaluate.py` identifies a limited set of COCO category names, several compounds, simple plural forms, and a small synonym list. `evaluate_caption()` compares those extracted names with caller-supplied object labels.

- `chair_s` is a per-caption binary value: 1 if at least one extracted object is absent from the supplied labels, otherwise 0.
- `chair_i` is the percentage of extracted mentions absent from the supplied labels, for that caption.
- Without labels, both are `null` and the status is `not_evaluated`.

This is explicitly labeled **CHAIR-inspired**, not validated canonical CHAIR. Its vocabulary is limited, normalization is rule-based, and its results depend on the completeness and naming of the supplied labels. Do not compare it directly with published canonical CHAIR results.

The dataset utility reports the arithmetic mean of per-image CHAIR-like values across annotated, successful records. It does not calculate a corpus-level canonical CHAIR score.

## Research evidence

No CaptionLens benchmark results are included in the code. The [historical completion report](version-1/audit/CaptionLens_Completion_Report.txt) states that no CaptionLens experiments were run. Literature claims and their verification notes are available in the [claim-verification log](version-1/audit/CaptionLens_Claim_Verification.csv) and [literature CSV](version-1/literature/CaptionLens_Literature_Yearwise_2021_2026.csv).
