import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from peft import LoraConfig, get_peft_model

class CustomCaptionDataset(Dataset):
    def __init__(self, samples: list, processor):
        self.samples = samples
        self.processor = processor

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
            max_length=64, 
            return_tensors="pt"
        )
        # Remove batch dimension added by processor
        return {k: v.squeeze(0) for k, v in encoding.items()}

def train_lora_caption_model(samples: list, epochs: int = 3, lr: float = 5e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training LoRA adapter on device: {device}")

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    base_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # Configure Low-Rank Adaptation (LoRA) for the text decoder
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["query", "value"],
        lora_dropout=0.05,
        bias="none"
    )
    
    model = get_peft_model(base_model, lora_config).to(device)
    model.print_trainable_parameters()

    dataset = CustomCaptionDataset(samples, processor)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            pixel_values = batch["pixel_values"].to(device)

            labels = input_ids.clone()
            labels[labels == processor.tokenizer.pad_token_id] = -100 # Mask padding tokens

            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

    # Save fine-tuned adapter weights
    model.save_pretrained("./fine_tuned_blip_lora")
    print("Fine-tuned LoRA weights successfully saved to './fine_tuned_blip_lora'.")

if __name__ == "__main__":
    print("LoRA Fine-tuning pipeline ready.")