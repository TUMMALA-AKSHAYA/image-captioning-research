import io
import time
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torchvision.models.detection as detection
import torchvision.transforms as T
from typing import Dict, Any, List

class ProductionCaptionPipeline:
    """
    Production Vision-Language Captioning & Spatial Grounding Pipeline.
    Combines pre-trained BLIP captioning with real-time spatial bounding box extraction.
    """
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        print(f"Loading pre-trained Vision-Language models on {self.device}...")
        
        # 1. Load BLIP Captioning Model
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
        self.model.eval()
        
        # 2. Load Object Detector for Spatial Grounding Tokens
        self.detector = detection.fasterrcnn_resnet50_fpn(weights=detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(self.device)
        self.detector.eval()

        # COCO Category Map for Object Labels
        self.coco_names = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet',
            'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
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
        
        # Load Image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        orig_w, orig_h = image.size
        
        # 1. Generate Natural Language Caption via BLIP
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(
            **inputs, 
            max_new_tokens=max_length,
            num_beams=3,
            repetition_penalty=1.2,
            no_repeat_ngram_size=2
        )
        caption_text = self.processor.decode(output_ids[0], skip_special_tokens=True)

        # 2. Extract Real Bounding Box Spatial Tokens via Faster R-CNN
        transform = T.Compose([T.ToTensor()])
        img_tensor = transform(image).to(self.device)
        detections = self.detector([img_tensor])[0]

        grounding_boxes: List[Dict[str, Any]] = []
        # Color palette for canvas overlays
        colors = ["#10b981", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"]

        # Filter detections above 0.5 confidence score
        box_idx = 0
        for i in range(len(detections['scores'])):
            score = detections['scores'][i].item()
            if score > 0.5 and box_idx < 8:
                label_id = detections['labels'][i].item()
                label_name = self.coco_names[label_id] if label_id < len(self.coco_names) else "object"
                box = detections['boxes'][i].cpu().numpy()
                
                # Scale box to canvas coordinate space (400x300)
                x1 = float(box[0] * (400.0 / orig_w))
                y1 = float(box[1] * (300.0 / orig_h))
                w = float((box[2] - box[0]) * (400.0 / orig_w))
                h = float((box[3] - box[1]) * (300.0 / orig_h))

                grounding_boxes.append({
                    "label": label_name,
                    "confidence": round(score, 3),
                    "x": round(x1, 1),
                    "y": round(y1, 1),
                    "w": round(w, 1),
                    "h": round(h, 1),
                    "color": colors[box_idx % len(colors)]
                })
                box_idx += 1
        
        latency_ms = round((time.time() - start_time) * 1000, 2)
        words = caption_text.split()

        return {
            "caption": caption_text,
            "word_count": len(words),
            "latency_ms": latency_ms,
            "device": str(self.device),
            "grounding_boxes": grounding_boxes
        }