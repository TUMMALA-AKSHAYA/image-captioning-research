import io
import time
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from typing import Dict, Any

class ProductionCaptionPipeline:
    """
    Production Vision-Language Captioning Pipeline using Pre-trained BLIP Weights.
    Runs locally on PyTorch (CPU / Apple Silicon Metal / CUDA).
    """
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        print("Loading pre-trained BLIP Vision-Language model weights...")
        
        # Load pre-trained vision processor and image-captioning model
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
        self.model.eval()
        
        self.vocab_size = self.model.config.text_config.vocab_size

    @torch.no_grad()
    def generate_caption(
        self, 
        image_bytes: bytes, 
        max_length: int = 50, 
        temperature: float = 0.7,
        top_k: int = 5
    ) -> Dict[str, Any]:
        start_time = time.time()
        
        # Load and convert image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Preprocess visual tokens
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        # Autoregressive generation pass
        output_ids = self.model.generate(
            **inputs, 
            max_new_tokens=max_length,
            num_beams=3,
            repetition_penalty=1.2,  # Prevents repeating words like "sari sari sari"
            no_repeat_ngram_size=2   # Ensures no two-word phrases repeat
        )
        
        # Decode output token IDs to natural language text
        caption_text = self.processor.decode(output_ids[0], skip_special_tokens=True)
        
        latency_ms = round((time.time() - start_time) * 1000, 2)
        words = caption_text.split()

        return {
            "caption": caption_text,
            "word_count": len(words),
            "latency_ms": latency_ms,
            "device": str(self.device)
        }