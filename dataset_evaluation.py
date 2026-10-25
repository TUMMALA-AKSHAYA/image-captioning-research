import os
import json
from typing import List, Dict, Any, Optional
from evaluate import ProductionEvaluator

class DatasetEvaluator:
    """
    Dataset-level evaluation module for CaptionLens.
    Validates dataset manifests (JSON/JSONL), streams evaluation records,
    and aggregates hallucination metrics (CHAIR_s, CHAIR_i) across splits.
    """
    def __init__(self, evaluator: Optional[ProductionEvaluator] = None):
        self.evaluator = evaluator or ProductionEvaluator()

    def validate_manifest(self, manifest_path: str) -> Dict[str, Any]:
        """
        Validates a dataset manifest file (JSON or JSONL) for required fields:
        image_path, caption / reference_captions, and optional ground_truth_objects.
        """
        if not os.path.exists(manifest_path):
            return {"valid": False, "error": f"Manifest file not found at '{manifest_path}'."}

        records = []
        errors = []
        
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                if manifest_path.endswith('.jsonl'):
                    for line_num, line in enumerate(f, 1):
                        if not line.strip():
                            continue
                        try:
                            record = json.loads(line)
                            records.append((line_num, record))
                        except json.JSONDecodeError as e:
                            errors.append(f"Line {line_num}: Invalid JSON -> {e}")
                else:
                    data = json.load(f)
                    if isinstance(data, list):
                        for idx, record in enumerate(data):
                            records.append((idx + 1, record))
                    else:
                        return {"valid": False, "error": "JSON root must be a list of records."}
        except Exception as e:
            return {"valid": False, "error": f"Failed to read manifest: {e}"}

        valid_count = 0
        for item_id, rec in records:
            if not isinstance(rec, dict):
                errors.append(f"Record {item_id}: Must be a JSON object.")
                continue
            
            # Check required fields
            if "image_path" not in rec:
                errors.append(f"Record {item_id}: Missing required field 'image_path'.")
            elif not os.path.exists(rec["image_path"]):
                errors.append(f"Record {item_id}: Image file not found at '{rec['image_path']}'.")
            
            if "caption" not in rec and "reference_captions" not in rec:
                errors.append(f"Record {item_id}: Must contain 'caption' or 'reference_captions'.")
            
            valid_count += 1

        return {
            "valid": len(errors) == 0,
            "total_records": len(records),
            "valid_records": valid_count,
            "errors": errors
        }

    def evaluate_dataset(self, manifest_path: str) -> Dict[str, Any]:
        """
        Runs dataset-level evaluation across all valid records in the manifest.
        """
        validation = self.validate_manifest(manifest_path)
        if not validation["valid"] and validation["valid_records"] == 0:
            return {"status": "error", "validation": validation}

        results = []
        attempted = 0
        successful = 0
        failed = 0
        evaluated_count = 0

        total_chair_s = 0.0
        total_chair_i = 0.0

        with open(manifest_path, 'r', encoding='utf-8') as f:
            lines = f.readlines() if manifest_path.endswith('.jsonl') else json.load(f)

        for rec in lines:
            if isinstance(rec, str):
                rec = json.loads(rec)
            
            attempted += 1
            image_path = rec.get("image_path")
            gt_objects = rec.get("ground_truth_objects", None)
            ref_caption = rec.get("caption", rec.get("reference_captions", [""])[0])

            if not image_path or not os.path.exists(image_path):
                failed += 1
                continue

            try:
                with open(image_path, "rb") as img_file:
                    image_bytes = img_file.read()
                
                # Perform evaluation using ProductionEvaluator
                eval_res = self.evaluator.evaluate_caption(ref_caption, ground_truth_objects=gt_objects)
                successful += 1

                if eval_res["evaluation_status"] == "evaluated":
                    evaluated_count += 1
                    total_chair_s += eval_res["chair_s"]
                    total_chair_i += eval_res["chair_i"]

                results.append({
                    "image_path": image_path,
                    "caption": ref_caption,
                    "evaluation": eval_res
                })
            except Exception as e:
                failed += 1
                results.append({
                    "image_path": image_path,
                    "error": str(e)
                })

        avg_chair_s = (total_chair_s / evaluated_count) if evaluated_count > 0 else None
        avg_chair_i = (total_chair_i / evaluated_count) if evaluated_count > 0 else None

        return {
            "status": "completed",
            "metrics_summary": {
                "attempted": attempted,
                "successful": successful,
                "failed": failed,
                "evaluated_count": evaluated_count,
                "aggregate_chair_s": avg_chair_s,
                "aggregate_chair_i": avg_chair_i
            },
                "detailed_results": results
        }