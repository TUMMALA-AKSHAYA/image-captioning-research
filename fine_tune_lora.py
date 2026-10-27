import argparse
import json
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from peft import LoraConfig, get_peft_model

class DenseCaptionDataset(Dataset):
    """
    Custom PyTorch Dataset for loading image-caption pairs (e.g., DOCCI or COCONut splits).
    """
    def __init__(self, samples: list, processor, max_length: int = 64):
        self.samples = samples
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        caption = item["caption"]
        
        encoding = self.processor(
            images=image, 
            text=caption, 
            padding="max_length", 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}

def train_lora_adapter(
    samples: list,
    epochs: int = 2,
    batch_size: int = 2,
    lr: float = 5e-5,
    output_dir: str = "./fine_tuned_blip_lora",
):
    """
    Executes Parameter-Efficient Fine-Tuning (LoRA) on BLIP's text decoder.
    """
    if not samples:
        raise ValueError("At least one real image-caption training sample is required.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing LoRA Fine-Tuning pipeline on device: {device}")

    model_id = "Salesforce/blip-image-captioning-base"
    
    print("Loading processor and base model from local cache...")
    try:
        processor = BlipProcessor.from_pretrained(model_id, local_files_only=True)
        base_model = BlipForConditionalGeneration.from_pretrained(model_id, local_files_only=True)
    except Exception as e:
        print(f"Local cache load failed ({e}), attempting online download...")
        processor = BlipProcessor.from_pretrained(model_id)
        base_model = BlipForConditionalGeneration.from_pretrained(model_id)

    # Configure Low-Rank Adaptation (LoRA) for text decoder query and value layers
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["query", "value"],
        lora_dropout=0.05,
        bias="none"
    )
    
    model = get_peft_model(base_model, lora_config).to(device)
    model.print_trainable_parameters()

    dataset = DenseCaptionDataset(samples, processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            pixel_values = batch["pixel_values"].to(device)

            labels = input_ids.clone()
            labels[labels == processor.tokenizer.pad_token_id] = -100

            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(dataloader))
        print(f"Epoch [{epoch+1}/{epochs}] - Training Loss: {avg_loss:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    print(f"LoRA adapter weights successfully saved to '{output_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BLIP with LoRA from a JSON/JSONL manifest.")
    parser.add_argument("manifest", help="JSON list or JSONL file with image_path and caption fields")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--output-dir", default="./fine_tuned_blip_lora")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    if manifest_path.suffix.lower() == ".jsonl":
        with manifest_path.open(encoding="utf-8") as manifest_file:
            samples = [json.loads(line) for line in manifest_file if line.strip()]
    else:
        with manifest_path.open(encoding="utf-8") as manifest_file:
            samples = json.load(manifest_file)
    if not isinstance(samples, list):
        parser.error("The JSON manifest root must be a list of samples.")
    if not samples:
        parser.error("The training manifest must contain at least one sample.")

    for sample in samples:
        if not isinstance(sample, dict) or not isinstance(sample.get("caption"), str):
            parser.error("Each sample must be an object with a string 'caption'.")
        image_path = sample.get("image_path")
        if not isinstance(image_path, str) or not image_path:
            parser.error("Each sample must have a non-empty 'image_path'.")
        resolved_image = Path(image_path)
        if not resolved_image.is_absolute():
            resolved_image = manifest_path.parent / resolved_image
        if not resolved_image.is_file():
            parser.error(f"Training image does not exist: {resolved_image}")
        sample["image_path"] = str(resolved_image)

    train_lora_adapter(
        samples=samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        output_dir=args.output_dir,
    )
