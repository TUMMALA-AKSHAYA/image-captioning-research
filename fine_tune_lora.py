import os
import io
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

def train_lora_adapter(samples: list, epochs: int = 2, batch_size: int = 2, lr: float = 5e-5):
    """
    Executes Parameter-Efficient Fine-Tuning (LoRA) on BLIP's text decoder.
    """
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

    # Generate verification dummy sample if none provided
    if not samples:
        print("No training samples provided. Generating verification dummy sample...")
        dummy_img_path = "temp_test_image.jpg"
        Image.new("RGB", (224, 224), color="blue").save(dummy_img_path)
        samples = [{"image_path": dummy_img_path, "caption": "a blue test image for verification"}]

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

    output_dir = "./fine_tuned_blip_lora"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    print(f"LoRA adapter weights successfully saved to '{output_dir}'.")

if __name__ == "__main__":
    train_lora_adapter(samples=[])