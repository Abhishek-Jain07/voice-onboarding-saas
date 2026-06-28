"""
Profile Aggregation Service
- Merge outputs from all AI services
- Compute cross-service confidence
- Create unified structured profile
"""

from __future__ import annotations

from typing import Any

from app.services.base import BaseService


class ProfileAggregationService(BaseService):
    name = "profile_aggregation"

    # ── Core Processing ─────────────────────────────────────────────────

    async def _process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        Merge all service outputs into a unified profile.
        input_data keys expected:
            - audio_features
            - emotion
            - sentiment
            - interests
            - personality
            - keywords_entities
            - conversation_summary
            - normalized_transcript
        """

        audio_features = input_data.get("audio_features", {})
        emotion = input_data.get("emotion", {})
        sentiment = input_data.get("sentiment", {})
        interests = input_data.get("interests", {})
        personality = input_data.get("personality", {})
        keywords_entities = input_data.get("keywords_entities", {})
        conversation_summary = input_data.get("conversation_summary", {})
        normalized = input_data.get("normalized_transcript", {})

        # Compute cross-service confidence
        confidences = []
        for service_result in [emotion, sentiment, interests, personality]:
            conf = service_result.get("confidence", 0.0)
            if conf:
                confidences.append(conf)

        # Weight audio features contribution
        if audio_features.get("voice_stability", 0) > 0:
            confidences.append(audio_features.get("voice_stability", 0.5))

        cross_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )

        # ── Validate and enrich ─────────────────────────────────────────

        # Cross-reference emotion with sentiment
        emotion_sentiment_agreement = self._check_agreement(emotion, sentiment)
        if emotion_sentiment_agreement:
            cross_confidence = min(cross_confidence + 0.1, 1.0)

        # Strip internal meta from sub-results
        for d in [audio_features, emotion, sentiment, interests,
                   personality, keywords_entities, conversation_summary, normalized]:
            d.pop("_meta", None)
            d.pop("fallback_reason", None)

        return {
            "audio_features": audio_features,
            "emotion": emotion,
            "sentiment": sentiment,
            "interests": interests,
            "personality": personality,
            "keywords_entities": keywords_entities,
            "conversation_summary": conversation_summary,
            "transcript": normalized,
            "cross_confidence": round(cross_confidence, 3),
        }

    @staticmethod
    def _check_agreement(emotion: dict, sentiment: dict) -> bool:
        """Check if emotion and sentiment broadly agree."""
        primary_emotion = emotion.get("primary_emotion", "neutral")
        sentiment_label = sentiment.get("overall_label", "NEUTRAL")

        positive_emotions = {"happy", "excited", "calm", "surprised"}
        negative_emotions = {"sad", "angry", "fearful"}

        if primary_emotion in positive_emotions and sentiment_label == "POSITIVE":
            return True
        if primary_emotion in negative_emotions and sentiment_label == "NEGATIVE":
            return True
        if primary_emotion == "neutral" and sentiment_label == "NEUTRAL":
            return True
        return False

    # ── Fallback ────────────────────────────────────────────────────────

    async def _fallback(self, input_data: dict[str, Any], error: Exception) -> dict[str, Any]:
        return {
            "audio_features": input_data.get("audio_features", {}),
            "emotion": input_data.get("emotion", {}),
            "sentiment": input_data.get("sentiment", {}),
            "interests": input_data.get("interests", {}),
            "personality": input_data.get("personality", {}),
            "keywords_entities": input_data.get("keywords_entities", {}),
            "conversation_summary": input_data.get("conversation_summary", {}),
            "transcript": input_data.get("normalized_transcript", {}),
            "cross_confidence": 0.0,
            "fallback_reason": str(error),
        }
