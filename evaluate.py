import time
import json
import torch
from PIL import Image
from typing import List, Dict, Any
from transformers import BlipProcessor, BlipForConditionalGeneration

class ProductionEvaluator:
    def __init__(self, model_id: str = "Salesforce/blip-image-captioning-base", device: str = "cpu"):
        self.device = torch.device(device)
        print(f"Loading evaluation model '{model_id}' on {self.device}...")
        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(model_id).to(self.device)
        self.model.eval()

        # Standard COCO object list for CHAIR hallucination calculation
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
    def evaluate_dataset(self, dataset_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Runs batch evaluation over a dataset list containing image paths and ground-truth objects/captions.
        Calculates average latency, word length, and CHAIR hallucination percentages.
        """
        total_latency = 0.0
        total_words = 0
        hallucinated_sentence_count = 0
        total_sentences = len(dataset_samples)

        results = []

        for sample in dataset_samples:
            img_path = sample["image_path"]
            gt_objects = set(sample.get("ground_truth_objects", []))
            
            image = Image.open(img_path).convert("RGB")
            
            start = time.time()
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=50,
                num_beams=3,
                repetition_penalty=1.2,
                no_repeat_ngram_size=2
            )
            generated_text = self.processor.decode(output_ids[0], skip_special_tokens=True)
            latency = (time.time() - start) * 1000

            total_latency += latency
            words = generated_text.lower().split()
            total_words += len(words)

            # CHAIR Sentence-level hallucination calculation
            mentioned_objects = [w for w in words if w in self.coco_objects]
            hallucinated_in_caption = [obj for obj in mentioned_objects if obj not in gt_objects]

            if len(hallucinated_in_caption) > 0:
                hallucinated_sentence_count += 1

            results.append({
                "image_path": img_path,
                "generated_caption": generated_text,
                "latency_ms": round(latency, 2),
                "hallucinated_objects": hallucinated_in_caption
            })

        avg_latency = round(total_latency / max(1, total_sentences), 2)
        avg_words = round(total_words / max(1, total_sentences), 2)
        chair_sentence_score = round((hallucinated_sentence_count / max(1, total_sentences)) * 100, 2)

        return {
            "total_evaluated_samples": total_sentences,
            "average_latency_ms": avg_latency,
            "average_caption_length_words": avg_words,
            "CHAIR_s_hallucination_rate_pct": chair_sentence_score,
            "detailed_sample_results": results
        }

if __name__ == "__main__":
    evaluator = ProductionEvaluator()
    # Benchmark execution output log
    print("Evaluator initialized successfully. Ready for COCO / DOCCI validation splits.")