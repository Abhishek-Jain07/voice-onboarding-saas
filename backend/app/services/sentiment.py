"""
Sentiment Analysis Service
- HuggingFace transformers pipeline (distilbert-base-uncased-finetuned-sst-2-english)
- Overall sentiment + per-sentence breakdown
- Rule-based fallback using keyword matching
"""

from __future__ import annotations

import re
from typing import Any

from app.config import settings
from app.services.base import BaseService

try:
    from transformers import pipeline as hf_pipeline

    HAS_TRANSFORMERS = False
except ImportError:
    HAS_TRANSFORMERS = False


class SentimentService(BaseService):
    name = "sentiment"

    def __init__(self) -> None:
        super().__init__()
        self._pipeline = None
        if not HAS_TRANSFORMERS:
            self._fallback_mode = True
            self.logger.warning("transformers not installed – using rule-based sentiment")

    def _load_pipeline(self) -> None:
        if self._pipeline is not None:
            return
        if not HAS_TRANSFORMERS:
            raise RuntimeError("transformers not installed")
        self.logger.info("Loading sentiment model", extra={"service": self.name})
        self._pipeline = hf_pipeline(
            "sentiment-analysis",
            model=settings.sentiment_model,
            truncation=True,
            max_length=512,
        )

    # ── Core Processing ─────────────────────────────────────────────────

    async def _process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        self._load_pipeline()
        text: str = input_data.get("normalized_text", "") or input_data.get("text", "")
        sentences: list[str] = input_data.get("sentences", [])

        if not sentences:
            sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

        if not sentences:
            return {
                "overall_label": "NEUTRAL",
                "overall_score": 0.0,
                "sentences": [],
            }

        # Per-sentence analysis
        sentence_results = []
        for sent in sentences:
            if not sent.strip():
                continue
            result = self._pipeline(sent[:512])[0]
            sentence_results.append({
                "text": sent,
                "label": result["label"],
                "score": round(result["score"], 4),
            })

        # Overall: weighted average
        if sentence_results:
            pos_score = sum(
                r["score"] for r in sentence_results if r["label"] == "POSITIVE"
            )
            neg_score = sum(
                r["score"] for r in sentence_results if r["label"] == "NEGATIVE"
            )
            total = len(sentence_results)
            if pos_score > neg_score:
                overall_label = "POSITIVE"
                overall_score = pos_score / total
            elif neg_score > pos_score:
                overall_label = "NEGATIVE"
                overall_score = neg_score / total
            else:
                overall_label = "NEUTRAL"
                overall_score = 0.5
        else:
            overall_label = "NEUTRAL"
            overall_score = 0.0

        return {
            "overall_label": overall_label,
            "overall_score": round(overall_score, 4),
            "sentences": sentence_results,
        }

    # ── Fallback: rule-based ────────────────────────────────────────────

    async def _fallback(self, input_data: dict[str, Any], error: Exception) -> dict[str, Any]:
        text: str = input_data.get("normalized_text", "") or input_data.get("text", "")
        sentences: list[str] = input_data.get("sentences", [])
        if not sentences:
            sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

        positive_words = {
            "love", "enjoy", "great", "amazing", "wonderful", "happy", "excited",
            "good", "fantastic", "beautiful", "fun", "awesome", "kind", "warm",
            "best", "like", "appreciate", "grateful", "positive", "adventurous",
        }
        negative_words = {
            "hate", "dislike", "terrible", "awful", "bad", "sad", "angry",
            "worst", "horrible", "boring", "annoying", "frustrating", "ugly",
            "pain", "difficult", "problem", "worried", "scared", "stressed",
        }

        sentence_results = []
        for sent in sentences:
            if not sent.strip():
                continue
            words = set(sent.lower().split())
            pos_count = len(words & positive_words)
            neg_count = len(words & negative_words)

            if pos_count > neg_count:
                label = "POSITIVE"
                score = min(0.5 + pos_count * 0.1, 0.95)
            elif neg_count > pos_count:
                label = "NEGATIVE"
                score = min(0.5 + neg_count * 0.1, 0.95)
            else:
                label = "NEUTRAL"
                score = 0.5

            sentence_results.append({
                "text": sent,
                "label": label,
                "score": round(score, 4),
            })

        # Overall
        pos_total = sum(1 for r in sentence_results if r["label"] == "POSITIVE")
        neg_total = sum(1 for r in sentence_results if r["label"] == "NEGATIVE")
        total = max(len(sentence_results), 1)

        if pos_total > neg_total:
            overall_label = "POSITIVE"
            overall_score = pos_total / total
        elif neg_total > pos_total:
            overall_label = "NEGATIVE"
            overall_score = neg_total / total
        else:
            overall_label = "NEUTRAL"
            overall_score = 0.5

        return {
            "overall_label": overall_label,
            "overall_score": round(overall_score, 4),
            "sentences": sentence_results,
            "fallback_reason": str(error),
        }
