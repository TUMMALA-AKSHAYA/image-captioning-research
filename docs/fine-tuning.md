# BLIP LoRA fine-tuning

`fine_tune_lora.py` defines `DenseCaptionDataset` and `train_lora_adapter()`. The CLI requires a real JSON or JSONL manifest; it no longer silently substitutes a synthetic image. A record must contain `image_path` and a string `caption`. Relative image paths are resolved relative to the manifest file.

Example JSON:

```json
[
  {"image_path":"images/0001.jpg","caption":"A person riding a bicycle."}
]
```

Example JSONL has one such object per line. Start training with:

```bash
python fine_tune_lora.py data/train.json \
  --epochs 2 --batch-size 2 --learning-rate 0.00005 \
  --output-dir ./fine_tuned_blip_lora
```

The script loads pretrained `Salesforce/blip-image-captioning-base`, applies PEFT LoRA to modules named `query` and `value` (rank 8, alpha 32, dropout 0.05), and saves adapter weights to the requested directory. This is fine-tuning, not training from scratch.

The inference API currently loads the base BLIP checkpoint and does **not** load the saved adapter. Training a checkpoint is not evidence of a completed experiment or improved quality; use a fixed train/validation/test protocol and evaluate the adapter separately before making claims. Checkpoint directories are ignored by Git by default.

See [cloud compute notes](cloud-training.md) for device behavior and [dataset evaluation](dataset-evaluation.md) for a separate evaluation manifest.
