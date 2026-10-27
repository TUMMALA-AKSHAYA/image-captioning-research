import io
import time
import re
import torch
from PIL import Image
from typing import List, Dict, Any, Optional
from transformers import BlipProcessor, BlipForConditionalGeneration

class ProductionEvaluator:
    """
    Production-grade COCO CHAIR Object Hallucination Evaluation Engine.
    Distinguishes sentence-level (CHAIR_s) and instance-level (CHAIR_i) scores,
    handling missing ground-truth annotations with explicit null/not_evaluated semantics.
    """
    def __init__(self, model_id: str = "Salesforce/blip-image-captioning-base", device: str = "cpu"):
        self.device = torch.device(device)
        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(model_id).to(self.device)
        self.model.eval()

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

        self.synonyms = {
            "guy": "person", "man": "person", "woman": "person", "boy": "person", "girl": "person",
            "player": "person", "child": "person", "lady": "person", "gentleman": "person",
            "bike": "bicycle", "automobile": "car", "puppy": "dog", "kitty": "cat"
        }

    @torch.no_grad()
    def generate_caption(self, image_bytes: bytes) -> str:
        """Generate a caption for one image using this evaluator's BLIP model."""
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=50,
            num_beams=3,
            repetition_penalty=1.2,
        )
        return self.processor.decode(output_ids[0], skip_special_tokens=True)

    def normalize_and_extract_objects(self, caption: str) -> List[str]:
        caption_lower = caption.lower()
        compounds = [
            "hot dog", "sports ball", "fire hydrant", "traffic light", "stop sign", 
            "parking meter", "baseball bat", "baseball glove", "tennis racket", 
            "wine glass", "potted plant", "dining table", "cell phone", "teddy bear", "hair drier"
        ]
        
        found_objects = []
        for compound in compounds:
            if compound in caption_lower:
                found_objects.append(compound)
                caption_lower = caption_lower.replace(compound, " [FOUND] ")

        tokens = re.findall(r'\b[a-z]+\b', caption_lower)
        for token in tokens:
            if token == "found":
                continue
            singular = token[:-1] if token.endswith('s') and not token.endswith('ss') else token
            mapped = self.synonyms.get(singular, singular)
            if mapped in self.coco_objects:
                found_objects.append(mapped)

        return found_objects

    def evaluate_caption(
        self, 
        caption: str, 
        ground_truth_objects: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates object hallucination for a given caption text against optional ground truth objects.
        Returns explicit 'not_evaluated' status if ground truth is absent.
        """
        mentioned_objects = self.normalize_and_extract_objects(caption)

        if ground_truth_objects is None:
            return {
                "evaluation_status": "not_evaluated",
                "evaluation_method": "CHAIR-inspired COCO normalized (Pending Ground Truth)",
                "ground_truth_provided": False,
                "mentioned_objects": mentioned_objects,
                "hallucinated_objects": [],
                "chair_s": None,
                "chair_i": None,
                "message": "Ground-truth object annotations not provided; hallucination score not computed."
            }

        gt_set = set([obj.lower() for obj in ground_truth_objects])
        hallucinated_objects = [obj for obj in mentioned_objects if obj not in gt_set]
        
        has_hallucination = len(hallucinated_objects) > 0 if mentioned_objects else False
        chair_i = (len(hallucinated_objects) / len(mentioned_objects) * 100.0) if mentioned_objects else 0.0

        return {
            "evaluation_status": "evaluated",
            "evaluation_method": "CHAIR-inspired COCO normalized",
            "ground_truth_provided": True,
            "mentioned_objects": mentioned_objects,
            "hallucinated_objects": hallucinated_objects,
            "chair_s": 1.0 if has_hallucination else 0.0,
            "chair_i": round(chair_i, 2),
            "message": "Evaluation successfully completed against provided ground-truth objects."
        }
