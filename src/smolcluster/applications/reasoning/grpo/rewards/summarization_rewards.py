"""Reward functions for summarization tasks."""

from typing import Any, Optional

from rouge_score import rouge_scorer as _rouge_scorer

_scorer = _rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def calculate_summary_quality(predicted: str, reference: str) -> float:
    """ROUGE-L F1 between a predicted summary and the gold reference.

    Returns float in [0, 1]. Higher is better.
    Uses stemming so minor morphological variants don't get penalised.
    """
    if not predicted.strip() or not reference.strip():
        return 0.0
    scores = _scorer.score(reference, predicted)
    return float(scores["rougeL"].fmeasure)


def calculate_length_reward(
    predicted_answer: str,
    max_length: int,
    tokenizer: Optional[Any] = None,
) -> float:
    """
    Calculate a reward based on the length of the predicted answer.

    Uses token count when a tokenizer is provided, otherwise falls back to
    character count.

    Args:
        predicted_answer: The answer predicted by the model.
        max_length: Target length in tokens (or characters if no tokenizer).
        tokenizer: Optional tokenizer; unwraps mlx-lm's _tokenizer wrapper.
    Returns:
        A value in (-1, 0], where 0 means exactly max_length.
    """
    if tokenizer is not None:
        try:
            hf_tok = getattr(tokenizer, "_tokenizer", tokenizer)
            length = len(hf_tok.encode(predicted_answer, add_special_tokens=False))
        except Exception:
            logger.error("Failed to tokenize with provided tokenizer")
            
            return None
        
    return -((abs(length - max_length)) / max_length)
