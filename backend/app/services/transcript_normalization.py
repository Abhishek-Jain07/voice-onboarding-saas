"""
Transcript Normalization Service
- Filler word removal (um, uh, like, you know, …)
- Punctuation restoration
- Repeated word normalization
- Language detection
- Sentence segmentation
"""

from __future__ import annotations

import re
from typing import Any

from app.services.base import BaseService

FILLER_WORDS = {
    "um", "uh", "uhm", "umm", "hmm", "hm", "er", "ah",
    "like", "you know", "i mean", "sort of", "kind of",
    "basically", "actually", "literally", "right", "okay so",
    "well", "so yeah", "you see", "i guess",
}

# Pre-compile filler patterns (word-boundary safe)
_FILLER_PATTERNS = [
    re.compile(r"\b" + re.escape(f) + r"\b", re.IGNORECASE)
    for f in sorted(FILLER_WORDS, key=len, reverse=True)  # longest first
]


class TranscriptNormalizationService(BaseService):
    name = "transcript_normalization"

    # ── Core Processing ─────────────────────────────────────────────────

    async def _process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        text: str = input_data.get("text", "")
        language: str = input_data.get("language", "en")

        if not text.strip():
            return {
                "original_text": text,
                "normalized_text": "",
                "sentences": [],
                "filler_words_removed": 0,
                "language": language,
            }

        original_text = text

        # Step 1 – Remove filler words
        filler_count = 0
        for pattern in _FILLER_PATTERNS:
            matches = pattern.findall(text)
            filler_count += len(matches)
            text = pattern.sub("", text)

        # Step 2 – Normalize whitespace
        text = re.sub(r"\s{2,}", " ", text).strip()

        # Step 3 – Remove repeated words ("I I went" → "I went")
        text = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", text, flags=re.IGNORECASE)

        # Step 4 – Basic punctuation restoration
        text = self._restore_punctuation(text)

        # Step 5 – Sentence segmentation
        sentences = self._segment_sentences(text)

        # Step 6 – Capitalize first letter of each sentence
        sentences = [s[0].upper() + s[1:] if s else s for s in sentences]
        normalized_text = " ".join(sentences)

        return {
            "original_text": original_text,
            "normalized_text": normalized_text,
            "sentences": sentences,
            "filler_words_removed": filler_count,
            "language": language,
        }

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _restore_punctuation(text: str) -> str:
        """Add basic punctuation if missing."""
        # Ensure the text ends with a period
        text = text.strip()
        if text and text[-1] not in ".!?":
            text += "."

        # Add periods before common sentence starters when missing
        sentence_starters = [
            "I ", "My ", "We ", "He ", "She ", "They ", "It ",
            "The ", "This ", "That ", "And ", "But ", "So ", "Then ",
            "Also ", "However ", "Because ",
        ]
        for starter in sentence_starters:
            # If starter appears mid-sentence without preceding punctuation
            pattern = re.compile(
                r"([a-z])\s+(" + re.escape(starter.strip()) + r"\s)",
                re.IGNORECASE,
            )
            text = pattern.sub(r"\1. \2", text)

        return text

    @staticmethod
    def _segment_sentences(text: str) -> list[str]:
        """Split text into sentences."""
        # Split on sentence-ending punctuation
        raw = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in raw if s.strip()]
        return sentences

    # ── Fallback ────────────────────────────────────────────────────────

    async def _fallback(self, input_data: dict[str, Any], error: Exception) -> dict[str, Any]:
        text = input_data.get("text", "")
        return {
            "original_text": text,
            "normalized_text": text,
            "sentences": [text] if text else [],
            "filler_words_removed": 0,
            "language": input_data.get("language", "en"),
            "fallback_reason": str(error),
        }
