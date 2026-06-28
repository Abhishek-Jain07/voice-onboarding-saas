"""
Personality Analysis Service
- Big Five personality traits via zero-shot classification
- Communication style detection
- Attachment style estimation
- Rule-based fallback using linguistic patterns
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

# ── Label definitions ───────────────────────────────────────────────────

BIG_FIVE_LABELS = {
    "openness": [
        "creative and imaginative", "curious and open to new experiences",
        "artistic and unconventional",
    ],
    "conscientiousness": [
        "organized and disciplined", "responsible and dependable",
        "goal-oriented and hardworking",
    ],
    "extraversion": [
        "outgoing and sociable", "energetic and talkative",
        "enthusiastic and assertive",
    ],
    "agreeableness": [
        "kind and cooperative", "trusting and helpful",
        "empathetic and considerate",
    ],
    "neuroticism": [
        "anxious and worried", "emotionally sensitive",
        "self-conscious and vulnerable",
    ],
}

COMMUNICATION_STYLES = [
    "direct and assertive",
    "warm and empathetic",
    "analytical and precise",
    "storyteller and narrative",
    "humorous and playful",
    "reserved and thoughtful",
]

ATTACHMENT_STYLES = [
    "secure and comfortable with intimacy",
    "anxious and seeks reassurance",
    "avoidant and values independence",
    "disorganized and conflicted about closeness",
]


class PersonalityService(BaseService):
    name = "personality"

    def __init__(self) -> None:
        super().__init__()
        self._classifier = None
        if not HAS_TRANSFORMERS:
            self._fallback_mode = True
            self.logger.warning("transformers not installed – using rule-based personality")

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
            return self._default_result()

        truncated = text[:1024]

        # Big Five
        big_five = {}
        for trait, labels in BIG_FIVE_LABELS.items():
            result = self._classifier(truncated, candidate_labels=labels, multi_label=True)
            # Average the top scores for this trait
            avg_score = sum(result["scores"]) / len(result["scores"])
            big_five[trait] = round(avg_score, 3)

        # Communication style
        comm_result = self._classifier(truncated, candidate_labels=COMMUNICATION_STYLES)
        communication_style = comm_result["labels"][0].split(" and ")[0]

        # Attachment style
        attach_result = self._classifier(truncated, candidate_labels=ATTACHMENT_STYLES)
        attachment_style = attach_result["labels"][0].split(" and ")[0]

        # Confidence: average of top scores
        confidence = (comm_result["scores"][0] + attach_result["scores"][0]) / 2

        return {
            "big_five": big_five,
            "communication_style": communication_style,
            "attachment_style": attachment_style,
            "confidence": round(confidence, 3),
        }

    # ── Fallback ────────────────────────────────────────────────────────

    async def _fallback(self, input_data: dict[str, Any], error: Exception) -> dict[str, Any]:
        """Rule-based personality estimation from linguistic patterns."""
        text: str = (
            input_data.get("normalized_text", "") or input_data.get("text", "")
        ).lower()

        if not text.strip():
            result = self._default_result()
            result["fallback_reason"] = str(error)
            return result

        words = text.split()
        word_count = len(words)

        # ── Big Five heuristics ──────────────────────────────────────────
        big_five = {
            "openness": 0.5,
            "conscientiousness": 0.5,
            "extraversion": 0.5,
            "agreeableness": 0.5,
            "neuroticism": 0.5,
        }

        # Openness: creative/exploratory words
        openness_words = {"creative", "imagine", "explore", "art", "curious", "new", "travel", "adventure", "different", "unique"}
        openness_count = sum(1 for w in words if w in openness_words)
        big_five["openness"] = min(0.4 + openness_count * 0.08, 0.95)

        # Conscientiousness: goal/work words
        consc_words = {"work", "goal", "plan", "organize", "career", "discipline", "responsible", "achieve", "focused", "dedicated"}
        consc_count = sum(1 for w in words if w in consc_words)
        big_five["conscientiousness"] = min(0.4 + consc_count * 0.08, 0.95)

        # Extraversion: social/energy words
        extra_words = {"love", "friends", "party", "social", "fun", "together", "people", "enjoy", "exciting", "outgoing"}
        extra_count = sum(1 for w in words if w in extra_words)
        big_five["extraversion"] = min(0.4 + extra_count * 0.08, 0.95)

        # Agreeableness: empathy/kindness words
        agree_words = {"kind", "help", "care", "share", "support", "understand", "empathy", "warm", "gentle", "compassion"}
        agree_count = sum(1 for w in words if w in agree_words)
        big_five["agreeableness"] = min(0.4 + agree_count * 0.08, 0.95)

        # Neuroticism: anxiety/worry words
        neuro_words = {"worry", "anxious", "stressed", "nervous", "scared", "afraid", "overwhelmed", "insecure", "doubt", "tense"}
        neuro_count = sum(1 for w in words if w in neuro_words)
        big_five["neuroticism"] = min(0.3 + neuro_count * 0.1, 0.95)

        # Round
        big_five = {k: round(v, 3) for k, v in big_five.items()}

        # Communication style
        if "?" in text and text.count("?") > 2:
            communication_style = "analytical"
        elif word_count > 100:
            communication_style = "storyteller"
        elif any(w in text for w in ["haha", "lol", "funny", "joke"]):
            communication_style = "humorous"
        elif big_five["agreeableness"] > 0.6:
            communication_style = "warm"
        else:
            communication_style = "balanced"

        # Attachment style
        if big_five["agreeableness"] > 0.6 and big_five["neuroticism"] < 0.5:
            attachment_style = "secure"
        elif big_five["neuroticism"] > 0.6:
            attachment_style = "anxious"
        elif big_five["extraversion"] < 0.4:
            attachment_style = "avoidant"
        else:
            attachment_style = "secure"

        return {
            "big_five": big_five,
            "communication_style": communication_style,
            "attachment_style": attachment_style,
            "confidence": 0.45,
            "fallback_reason": str(error),
        }

    @staticmethod
    def _default_result() -> dict[str, Any]:
        return {
            "big_five": {
                "openness": 0.5,
                "conscientiousness": 0.5,
                "extraversion": 0.5,
                "agreeableness": 0.5,
                "neuroticism": 0.5,
            },
            "communication_style": "balanced",
            "attachment_style": "secure",
            "confidence": 0.0,
        }
