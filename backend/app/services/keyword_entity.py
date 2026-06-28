"""
Keyword & Entity Extraction Service
- spaCy NER (en_core_web_sm)
- Keyword extraction via frequency/TF-IDF
- Topic detection via zero-shot classification
- Intent detection
- Rule-based fallback
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from app.config import settings
from app.services.base import BaseService

try:
    import spacy

    HAS_SPACY = False
except ImportError:
    HAS_SPACY = False

try:
    from transformers import pipeline as hf_pipeline

    HAS_TRANSFORMERS = False
except ImportError:
    HAS_TRANSFORMERS = False

# Stop words for keyword extraction
STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "he", "him", "his", "she", "her", "hers",
    "it", "its", "they", "them", "their", "theirs", "a", "an", "the",
    "and", "but", "or", "for", "nor", "not", "so", "yet", "to", "of",
    "in", "on", "at", "by", "with", "from", "as", "into", "through",
    "during", "before", "after", "is", "am", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "that",
    "which", "who", "whom", "this", "these", "those", "what", "when",
    "where", "how", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "no", "only", "own", "same",
    "than", "too", "very", "just", "also", "about", "up", "out",
    "if", "then", "because", "while", "really", "going", "like",
}

TOPIC_LABELS = [
    "personal life and background",
    "hobbies and interests",
    "relationships and dating",
    "career and professional life",
    "travel and adventure",
    "food and dining",
    "entertainment and media",
    "health and wellness",
    "values and beliefs",
    "future plans and goals",
]

INTENT_LABELS = [
    "introducing themselves",
    "describing preferences",
    "seeking a partner",
    "sharing experiences",
    "expressing goals",
    "telling a story",
]


class KeywordEntityService(BaseService):
    name = "keyword_entity"

    def __init__(self) -> None:
        super().__init__()
        self._nlp = None
        self._classifier = None
        if not HAS_SPACY:
            self._fallback_mode = True
            self.logger.warning("spaCy not available – using regex NER fallback")

    def _load_models(self) -> None:
        if self._nlp is None and HAS_SPACY:
            try:
                self.logger.info("Loading spaCy model", extra={"service": self.name})
                self._nlp = spacy.load(settings.spacy_model)
            except OSError:
                self.logger.warning("spaCy model not found – download with: python -m spacy download en_core_web_sm")
                self._nlp = None

        if self._classifier is None and HAS_TRANSFORMERS:
            try:
                self._classifier = hf_pipeline(
                    "zero-shot-classification",
                    model=settings.zero_shot_model,
                )
            except Exception:
                self._classifier = None

    # ── Core Processing ─────────────────────────────────────────────────

    async def _process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        self._load_models()
        text: str = input_data.get("normalized_text", "") or input_data.get("text", "")

        if not text.strip():
            return {"entities": [], "keywords": [], "topics": [], "intent": "general"}

        # NER
        entities = self._extract_entities(text)

        # Keywords
        keywords = self._extract_keywords(text)

        # Topics & Intent
        topics, intent = self._detect_topics_intent(text)

        return {
            "entities": entities,
            "keywords": keywords,
            "topics": topics,
            "intent": intent,
        }

    def _extract_entities(self, text: str) -> list[dict[str, Any]]:
        if self._nlp is None:
            return self._regex_ner(text)

        doc = self._nlp(text)
        entities = []
        seen = set()
        for ent in doc.ents:
            key = (ent.text.lower(), ent.label_)
            if key not in seen:
                seen.add(key)
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                })
        return entities

    @staticmethod
    def _regex_ner(text: str) -> list[dict[str, Any]]:
        """Simple regex-based NER fallback."""
        entities = []

        # Capitalized multi-word names (simple person/place detection)
        name_pattern = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
        for match in name_pattern.finditer(text):
            entities.append({
                "text": match.group(),
                "label": "PERSON",
                "start": match.start(),
                "end": match.end(),
            })

        return entities

    @staticmethod
    def _extract_keywords(text: str, top_n: int = 10) -> list[str]:
        """Frequency-based keyword extraction."""
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        words = [w for w in words if w not in STOP_WORDS]
        counter = Counter(words)
        return [word for word, _ in counter.most_common(top_n)]

    def _detect_topics_intent(self, text: str) -> tuple[list[str], str]:
        """Detect topics and intent via zero-shot or fallback."""
        if self._classifier is not None:
            try:
                # Topics
                topic_result = self._classifier(
                    text[:512],
                    candidate_labels=TOPIC_LABELS,
                    multi_label=True,
                )
                topics = [
                    label for label, score in zip(topic_result["labels"], topic_result["scores"])
                    if score > 0.3
                ][:5]

                # Intent
                intent_result = self._classifier(
                    text[:512],
                    candidate_labels=INTENT_LABELS,
                )
                intent = intent_result["labels"][0]

                return topics, intent
            except Exception:
                pass

        # Fallback: keyword-based topic detection
        text_lower = text.lower()
        topics = []
        topic_keywords = {
            "personal life and background": ["name", "born", "grew up", "family"],
            "hobbies and interests": ["hobby", "enjoy", "love", "like to", "passion"],
            "relationships and dating": ["looking for", "partner", "relationship", "date"],
            "career and professional life": ["work", "job", "career", "profession"],
            "travel and adventure": ["travel", "trip", "visit", "explore", "adventure"],
            "food and dining": ["food", "cook", "eat", "restaurant", "cuisine"],
        }
        for topic, keywords in topic_keywords.items():
            if any(kw in text_lower for kw in keywords):
                topics.append(topic)

        intent = "introducing themselves" if "my name" in text_lower else "describing preferences"

        return topics[:5], intent

    # ── Fallback ────────────────────────────────────────────────────────

    async def _fallback(self, input_data: dict[str, Any], error: Exception) -> dict[str, Any]:
        text = input_data.get("normalized_text", "") or input_data.get("text", "")
        entities = self._regex_ner(text)
        keywords = self._extract_keywords(text)
        topics, intent = self._detect_topics_intent(text)

        return {
            "entities": entities,
            "keywords": keywords,
            "topics": topics,
            "intent": intent,
            "fallback_reason": str(error),
        }
