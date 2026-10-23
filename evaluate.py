import io
import time
import torch
from PIL import Image
from typing import List, Dict, Any
from transformers import BlipProcessor, BlipForConditionalGeneration

class ProductionEvaluator:
    """
    Automated Benchmark Evaluator for Caption Quality & CHAIR Hallucination Metrics.
    Calculates sentence-level (CHAIR_s) and instance-level (CHAIR_i) hallucination scores.
    """
    def __init__(self, model_id: str = "Salesforce/blip-image-captioning-base", device: str = "cpu"):
        self.device = torch.device(device)
        print(f"Loading evaluation pipeline on {self.device}...")
        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(model_id).to(self.device)
        self.model.eval()

        # COCO 80 Standard Objects
        self.coco_objects = set([
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
            "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ])

    @torch.no_grad()
    def evaluate_sample(self, image_bytes: bytes, ground_truth_objects: List[str]) -> Dict[str, Any]:
        start = time.time()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=50,
            num_beams=3,
            repetition_penalty=1.2,
            no_repeat_ngram_size=2
        )
        caption = self.processor.decode(output_ids[0], skip_special_tokens=True)
        latency_ms = round((time.time() - start) * 1000, 2)

        words = caption.lower().split()
        gt_set = set([obj.lower() for obj in ground_truth_objects])

        # Extract mentioned COCO objects
        mentioned_objects = [w for w in words if w in self.coco_objects]
        hallucinated_objects = [obj for obj in mentioned_objects if obj not in gt_set]

        has_hallucination = len(hallucinated_objects) > 0
        chair_i = (len(hallucinated_objects) / len(mentioned_objects) * 100) if mentioned_objects else 0.0

        return {
            "caption": caption,
            "latency_ms": latency_ms,
            "word_count": len(words),
            "mentioned_objects": mentioned_objects,
            "hallucinated_objects": hallucinated_objects,
            "chair_s_has_hallucination": has_hallucination,
            "chair_i_instance_score_pct": round(chair_i, 2)
        }