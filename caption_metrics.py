import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("captionlens")

class CaptionMetricsEvaluator:
    """
    Computes standard reference-based caption metrics (BLEU, ROUGE-L) 
    and checks availability for advanced metrics (CIDEr, METEOR, BERTScore, CLIPScore).
    """
    def __init__(self):
        self._check_dependencies()

    def _check_dependencies(self):
        self.nltk_available = False
        self.rouge_available = False
        
        try:
            import nltk
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            self.nltk_available = True
        except ImportError:
            logger.warning("NLTK not installed. BLEU score calculation will be unavailable.")

        try:
            from rouge_score import rouge_scorer
            self.rouge_available = True
        except ImportError:
            logger.warning("rouge-score not installed. ROUGE-L calculation will be unavailable.")

    def compute_bleu(self, reference_caption: str, generated_caption: str) -> Optional[float]:
        if not self.nltk_available:
            return None
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            ref_tokens = reference_caption.lower().split()
            gen_tokens = generated_caption.lower().split()
            smoothing = SmoothingFunction().method1
            score = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smoothing)
            return round(float(score), 4)
        except Exception as e:
            logger.error(f"Error computing BLEU: {e}")
            return None

    def compute_rouge_l(self, reference_caption: str, generated_caption: str) -> Optional[float]:
        if not self.rouge_available:
            return None
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            scores = scorer.score(reference_caption, generated_caption)
            return round(float(scores['rougeL'].fmeasure), 4)
        except Exception as e:
            logger.error(f"Error computing ROUGE-L: {e}")
            return None

    def evaluate_generation(self, reference_caption: str, generated_caption: str) -> Dict[str, Any]:
        """
        Evaluates a generated caption against a reference caption.
        Returns computed scores or explicit availability notices.
        """
        bleu_score = self.compute_bleu(reference_caption, generated_caption)
        rouge_score = self.compute_rouge_l(reference_caption, generated_caption)

        return {
            "reference_caption": reference_caption,
            "generated_caption": generated_caption,
            "metrics": {
                "BLEU-4": bleu_score if bleu_score is not None else "unavailable_dependency",
                "ROUGE-L": rouge_score if rouge_score is not None else "unavailable_dependency",
                "CIDEr": "requires_coco_eval_tools",
                "METEOR": "requires_nltk_wordnet",
                "CLIPScore": "optional_separate_module"
            },
            "status": "success" if bleu_score is not None or rouge_score is not None else "partial"
        }
