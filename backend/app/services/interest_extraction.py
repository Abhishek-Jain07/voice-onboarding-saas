"""
Interest Extraction Service
- Zero-shot classification (facebook/bart-large-mnli)
- Candidate labels for hobbies, movies, food, travel, sports, career, lifestyle
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

# ── Category definitions ────────────────────────────────────────────────

INTEREST_CATEGORIES: dict[str, list[str]] = {
    "hobbies": [
        "hiking", "reading", "gaming", "cooking", "photography", "painting",
        "gardening", "dancing", "yoga", "meditation", "crafts", "music",
        "writing", "fishing", "cycling", "running", "swimming", "gym",
        "volunteering", "board games",
    ],
    "movies_tv": [
        "sci-fi", "comedy", "drama", "action", "horror", "romance",
        "documentary", "anime", "thriller", "fantasy", "netflix", "marvel",
        "star wars", "sitcom",
    ],
    "food_drink": [
        "italian food", "sushi", "mexican food", "vegan", "vegetarian",
        "coffee", "wine", "craft beer", "baking", "barbecue", "thai food",
        "indian food", "chinese food", "french cuisine", "pizza", "pasta",
    ],
    "travel": [
        "beach", "mountains", "backpacking", "road trip", "europe",
        "asia", "camping", "adventure travel", "luxury travel",
        "cultural tourism", "solo travel", "island hopping",
    ],
    "sports": [
        "football", "basketball", "soccer", "tennis", "baseball",
        "volleyball", "golf", "martial arts", "surfing", "skiing",
        "snowboarding", "rock climbing", "skateboarding",
    ],
    "career": [
        "software engineering", "medicine", "teaching", "business",
        "marketing", "design", "finance", "law", "science", "research",
        "entrepreneurship", "freelancing", "creative industry",
    ],
    "lifestyle": [
        "fitness", "wellness", "minimalism", "sustainability",
        "fashion", "pets", "social life", "family oriented",
        "night life", "outdoor lifestyle", "urban living",
    ],
}

ZERO_SHOT_LABELS = [
    "hobbies and recreation",
    "movies and entertainment",
    "food and cooking",
    "travel and adventure",
    "sports and fitness",
    "career and work",
    "lifestyle and values",
    "relationships and dating",
    "music and arts",
    "technology and science",
]


class InterestExtractionService(BaseService):
    name = "interest_extraction"

    def __init__(self) -> None:
        super().__init__()
        self._classifier = None
        if not HAS_TRANSFORMERS:
            self._fallback_mode = True
            self.logger.warning("transformers not installed – using keyword matching")

    def _load_classifier(self) -> None:
        if self._classifier is not None:
            return
        if not HAS_TRANSFORMERS:
            raise RuntimeError("transformers not installed")
        self.logger.info("Loading zero-shot classifier", extra={"service": self.name})
        self._classifier = hf_pipeline(
            "zero-shot-classification",
            model=settings.zero_shot_model,
        )

    # ── Core Processing ─────────────────────────────────────────────────

    async def _process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        self._load_classifier()
        text: str = input_data.get("normalized_text", "") or input_data.get("text", "")

        if not text.strip():
            return {"interests": [], "raw_labels": {}}

        # Run zero-shot classification
        result = self._classifier(
            text[:1024],
            candidate_labels=ZERO_SHOT_LABELS,
            multi_label=True,
        )

        raw_labels = dict(zip(result["labels"], result["scores"]))

        # Extract specific items via keyword matching within the text
        interests = []
        text_lower = text.lower()

        for category, items in INTEREST_CATEGORIES.items():
            found = [item for item in items if item.lower() in text_lower]
            # Map category to zero-shot label score
            label_map = {
                "hobbies": "hobbies and recreation",
                "movies_tv": "movies and entertainment",
                "food_drink": "food and cooking",
                "travel": "travel and adventure",
                "sports": "sports and fitness",
                "career": "career and work",
                "lifestyle": "lifestyle and values",
            }
            zs_label = label_map.get(category, "")
            confidence = raw_labels.get(zs_label, 0.0)

            if found or confidence > 0.3:
                interests.append({
                    "category": category,
                    "items": found if found else [],
                    "confidence": round(confidence, 3),
                })

        return {
            "interests": interests,
            "raw_labels": {k: round(v, 4) for k, v in raw_labels.items()},
        }

    # ── Fallback ────────────────────────────────────────────────────────

    async def _fallback(self, input_data: dict[str, Any], error: Exception) -> dict[str, Any]:
        """Pure keyword-based interest extraction."""
        text: str = input_data.get("normalized_text", "") or input_data.get("text", "")
        text_lower = text.lower()

        interests = []
        raw_labels: dict[str, float] = {}

        for category, items in INTEREST_CATEGORIES.items():
            found = [item for item in items if item.lower() in text_lower]
            if found:
                confidence = min(0.3 + len(found) * 0.15, 0.9)
                interests.append({
                    "category": category,
                    "items": found,
                    "confidence": round(confidence, 3),
                })
                raw_labels[category] = confidence

        return {
            "interests": interests,
            "raw_labels": raw_labels,
            "fallback_reason": str(error),
        }
