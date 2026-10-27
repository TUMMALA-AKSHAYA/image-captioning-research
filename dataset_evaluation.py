import json
from pathlib import Path
from typing import Any, Optional

from evaluate import ProductionEvaluator


class DatasetEvaluator:
    """Run BLIP captioning and CHAIR-inspired checks over a JSON/JSONL manifest."""

    def __init__(self, evaluator: Optional[ProductionEvaluator] = None):
        self.evaluator = evaluator

    @staticmethod
    def _read_records(manifest_path: str) -> tuple[list[tuple[int, Any]], list[str], str | None]:
        path = Path(manifest_path)
        if not path.is_file():
            return [], [], f"Manifest file not found at '{manifest_path}'."

        records: list[tuple[int, Any]] = []
        errors: list[str] = []
        try:
            with path.open("r", encoding="utf-8") as manifest:
                if path.suffix.lower() == ".jsonl":
                    for line_num, line in enumerate(manifest, 1):
                        if not line.strip():
                            continue
                        try:
                            records.append((line_num, json.loads(line)))
                        except json.JSONDecodeError as exc:
                            errors.append(f"Line {line_num}: Invalid JSON ({exc.msg}).")
                else:
                    data = json.load(manifest)
                    if not isinstance(data, list):
                        return [], [], "JSON root must be a list of records."
                    records = list(enumerate(data, 1))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            return [], [], f"Failed to read manifest: {exc}"
        return records, errors, None

    def _validated_records(self, manifest_path: str):
        records, errors, file_error = self._read_records(manifest_path)
        if file_error:
            return [], {"valid": False, "error": file_error}

        valid_records = []
        base_dir = Path(manifest_path).resolve().parent
        for item_id, record in records:
            record_errors = []
            if not isinstance(record, dict):
                errors.append(f"Record {item_id}: Must be a JSON object.")
                continue

            image_path = record.get("image_path")
            if not isinstance(image_path, str) or not image_path.strip():
                record_errors.append(f"Record {item_id}: 'image_path' must be a non-empty string.")
            else:
                resolved_path = Path(image_path)
                if not resolved_path.is_absolute():
                    resolved_path = base_dir / resolved_path
                if not resolved_path.is_file():
                    record_errors.append(f"Record {item_id}: Image file not found at '{image_path}'.")
                else:
                    record = {**record, "image_path": str(resolved_path)}

            ground_truth = record.get("ground_truth_objects")
            if ground_truth is not None and (
                not isinstance(ground_truth, list)
                or any(not isinstance(item, str) for item in ground_truth)
            ):
                record_errors.append(
                    f"Record {item_id}: 'ground_truth_objects' must be a list of strings when provided."
                )

            if record_errors:
                errors.extend(record_errors)
            else:
                valid_records.append(record)

        validation = {
            "valid": not errors,
            "total_records": len(records),
            "valid_records": len(valid_records),
            "errors": errors,
        }
        return valid_records, validation

    def validate_manifest(self, manifest_path: str) -> dict[str, Any]:
        """Validate image paths and optional ground-truth object lists."""
        _, validation = self._validated_records(manifest_path)
        return validation

    def evaluate_dataset(self, manifest_path: str) -> dict[str, Any]:
        """Generate captions and aggregate object-hallucination checks per image."""
        records, validation = self._validated_records(manifest_path)
        if not records:
            return {"status": "error", "validation": validation}

        evaluator = self.evaluator or ProductionEvaluator()

        results = []
        attempted = len(records)
        successful = 0
        failed = 0
        evaluated_count = 0
        total_chair_s = 0.0
        total_chair_i = 0.0

        for record in records:
            image_path = record["image_path"]
            try:
                with open(image_path, "rb") as image_file:
                    image_bytes = image_file.read()

                caption = evaluator.generate_caption(image_bytes)
                evaluation = evaluator.evaluate_caption(
                    caption,
                    ground_truth_objects=record.get("ground_truth_objects"),
                )
                successful += 1
                if evaluation["evaluation_status"] == "evaluated":
                    evaluated_count += 1
                    total_chair_s += evaluation["chair_s"]
                    total_chair_i += evaluation["chair_i"]
                results.append({
                    "image_path": image_path,
                    "generated_caption": caption,
                    "evaluation": evaluation,
                })
            except Exception as exc:
                failed += 1
                results.append({"image_path": image_path, "error": str(exc)})

        return {
            "status": "completed",
            "validation": validation,
            "metrics_summary": {
                "attempted": attempted,
                "successful": successful,
                "failed": failed,
                "evaluated_count": evaluated_count,
                "mean_per_image_chair_s": total_chair_s / evaluated_count if evaluated_count else None,
                "mean_per_image_chair_i_pct": total_chair_i / evaluated_count if evaluated_count else None,
            },
            "detailed_results": results,
        }
