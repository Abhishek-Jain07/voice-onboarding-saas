"""
Profile Aggregation Service (v2 – Per-Question Aggregation)
─────────────────────────────────────────────────────────────
Accepts a list of per-question analysis results and computes:
- Weighted mean, median, std dev for numerical metrics
- Emotion distribution across all answers
- Aggregated personality traits (weighted by question importance)
- Deduplicated ranked interest list
- Consistency scores
- Aggregation statistics for the frontend dashboard
"""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any

from app.services.base import BaseService


class ProfileAggregationService(BaseService):
    name = "profile_aggregation"

    # ── Core Processing ─────────────────────────────────────────────────

    async def _process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        Aggregate per-question analysis results into a unified profile.

        Input:
            question_analyses: list[dict] — each with emotion, sentiment,
                audio_features, interests, personality, keywords_entities
            normalized_transcript: dict
            conversation_text: str
        """
        question_analyses = input_data.get("question_analyses", [])
        normalized = input_data.get("normalized_transcript", {})

        if not question_analyses:
            return self._empty_profile(normalized)

        # ── Aggregate Emotions ──────────────────────────────────────
        emotion_agg = self._aggregate_emotions(question_analyses)

        # ── Aggregate Sentiment ─────────────────────────────────────
        sentiment_agg = self._aggregate_sentiment(question_analyses)

        # ── Aggregate Audio Features ────────────────────────────────
        audio_agg = self._aggregate_audio_features(question_analyses)

        # ── Aggregate Personality (Big Five) ────────────────────────
        personality_agg = self._aggregate_personality(question_analyses)

        # ── Aggregate Interests ─────────────────────────────────────
        interests_agg = self._aggregate_interests(question_analyses)

        # ── Aggregate Keywords & Entities ───────────────────────────
        keywords_agg = self._aggregate_keywords(question_analyses)

        # ── Compute Cross-Service Confidence ────────────────────────
        confidences = []
        if emotion_agg.get("avg_confidence"):
            confidences.append(emotion_agg["avg_confidence"])
        if sentiment_agg.get("avg_score"):
            confidences.append(abs(sentiment_agg["avg_score"]))
        if audio_agg.get("voice_stability"):
            confidences.append(audio_agg["voice_stability"])

        cross_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )

        # ── Aggregation Statistics (for frontend dashboard) ─────────
        aggregation_stats = {
            "emotion_distribution": emotion_agg.get("distribution", {}),
            "avg_confidence": round(emotion_agg.get("avg_confidence", 0), 3),
            "avg_sentiment_score": round(sentiment_agg.get("avg_score", 0), 3),
            "personality_weighted": personality_agg.get("big_five", {}),
            "consistency_score": round(
                self._compute_consistency(question_analyses), 3
            ),
            "total_questions_detected": len(question_analyses),
            "total_questions_possible": 35,
        }

        # Strip meta from sub-results
        normalized_clean = dict(normalized)
        normalized_clean.pop("_meta", None)
        normalized_clean.pop("fallback_reason", None)

        return {
            "audio_features": audio_agg,
            "emotion": emotion_agg,
            "sentiment": sentiment_agg,
            "interests": interests_agg,
            "personality": personality_agg,
            "keywords_entities": keywords_agg,
            "transcript": normalized_clean,
            "cross_confidence": round(cross_confidence, 3),
            "aggregation_stats": aggregation_stats,
        }

    # ── Emotion Aggregation ──────────────────────────────────────────────

    def _aggregate_emotions(self, analyses: list[dict]) -> dict:
        """Aggregate emotions into a distribution + primary emotion."""
        emotion_counter: Counter = Counter()
        confidences = []
        emotion_timeline = []

        for qa in analyses:
            emo = qa.get("emotion", {})
            primary = emo.get("primary_emotion", "")
            conf = emo.get("confidence", 0.0)

            if primary:
                emotion_counter[primary] += 1
                confidences.append(conf)
                emotion_timeline.append({
                    "question_id": qa.get("question_id", 0),
                    "emotion": primary,
                    "confidence": round(conf, 3),
                })

        total = sum(emotion_counter.values()) or 1
        distribution = {
            k: round(v / total, 3) for k, v in emotion_counter.most_common()
        }

        primary_emotion = emotion_counter.most_common(1)[0][0] if emotion_counter else "Neutral"
        secondary_emotion = (
            emotion_counter.most_common(2)[1][0]
            if len(emotion_counter) > 1 else "n/a"
        )

        avg_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )

        return {
            "primary_emotion": primary_emotion,
            "secondary_emotion": secondary_emotion,
            "distribution": distribution,
            "avg_confidence": round(avg_confidence, 3),
            "timeline": emotion_timeline,
            "confidence": round(avg_confidence, 3),
        }

    # ── Sentiment Aggregation ────────────────────────────────────────────

    def _aggregate_sentiment(self, analyses: list[dict]) -> dict:
        """Aggregate sentiment scores using weighted mean."""
        scores = []
        weights = []
        sentiment_timeline = []

        for qa in analyses:
            sent = qa.get("sentiment", {})
            score = sent.get("overall_score", sent.get("score", 0.0))
            weight = qa.get("question_weight", 1.0)

            if score is not None and isinstance(score, (int, float)):
                scores.append(score)
                weights.append(weight)
                sentiment_timeline.append({
                    "question_id": qa.get("question_id", 0),
                    "label": sent.get("overall_label", sent.get("sentiment", "NEUTRAL")),
                    "score": round(score, 3),
                })

        if not scores:
            return {
                "overall_label": "NEUTRAL",
                "overall_score": 0.0,
                "avg_score": 0.0,
                "timeline": [],
            }

        # Weighted average
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        total_weight = sum(weights) or 1
        avg_score = weighted_sum / total_weight

        # Determine label
        if avg_score > 0.3:
            label = "POSITIVE"
        elif avg_score < -0.3:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"

        return {
            "overall_label": label,
            "overall_score": round(avg_score, 3),
            "avg_score": round(avg_score, 3),
            "std_dev": round(statistics.stdev(scores), 3) if len(scores) > 1 else 0.0,
            "timeline": sentiment_timeline,
        }

    # ── Audio Features Aggregation ───────────────────────────────────────

    def _aggregate_audio_features(self, analyses: list[dict]) -> dict:
        """Average audio features across all answers."""
        features_keys = [
            "speaking_rate_sps", "pitch_mean", "pitch_std", "energy_mean",
            "voice_stability", "pause_count", "pause_duration_avg",
            "speech_to_silence_ratio",
        ]

        sums: dict[str, list[float]] = {k: [] for k in features_keys}

        for qa in analyses:
            af = qa.get("audio_features", {})
            for key in features_keys:
                val = af.get(key)
                if val is not None and isinstance(val, (int, float)):
                    sums[key].append(val)

        result = {}
        for key, values in sums.items():
            if values:
                result[key] = round(sum(values) / len(values), 3)
            else:
                result[key] = 0.0

        # Determine speaking style
        rate = result.get("speaking_rate_sps", 0)
        if rate > 4.0:
            result["speaking_style"] = "Fast"
        elif rate > 2.5:
            result["speaking_style"] = "Moderate"
        else:
            result["speaking_style"] = "Slow"

        return result

    # ── Personality Aggregation ──────────────────────────────────────────

    def _aggregate_personality(self, analyses: list[dict]) -> dict:
        """Weighted average of Big Five traits across all answers."""
        traits = ["openness", "conscientiousness", "extraversion",
                  "agreeableness", "neuroticism"]

        trait_scores: dict[str, list[tuple[float, float]]] = {
            t: [] for t in traits
        }

        for qa in analyses:
            personality = qa.get("personality", {})
            big_five = personality.get("big_five", personality)
            weight = qa.get("question_weight", 1.0)

            for trait in traits:
                val = big_five.get(trait)
                if val is not None and isinstance(val, (int, float)):
                    trait_scores[trait].append((val, weight))

        big_five_result = {}
        for trait, score_weight_pairs in trait_scores.items():
            if score_weight_pairs:
                weighted_sum = sum(s * w for s, w in score_weight_pairs)
                total_weight = sum(w for _, w in score_weight_pairs) or 1
                big_five_result[trait] = round(weighted_sum / total_weight, 3)
            else:
                big_five_result[trait] = 0.5  # Default neutral

        # Communication style inference
        extraversion = big_five_result.get("extraversion", 0.5)
        if extraversion > 0.65:
            comm_style = "Expressive and outgoing"
        elif extraversion < 0.35:
            comm_style = "Thoughtful and reserved"
        else:
            comm_style = "Balanced communicator"

        return {
            "big_five": big_five_result,
            "communication_style": comm_style,
        }

    # ── Interest Aggregation ─────────────────────────────────────────────

    def _aggregate_interests(self, analyses: list[dict]) -> dict:
        """Merge and deduplicate interests from all answers."""
        all_interests: Counter = Counter()
        all_hobbies: Counter = Counter()
        all_values: Counter = Counter()

        for qa in analyses:
            interests = qa.get("interests", {})
            for item in interests.get("interests", []):
                if isinstance(item, str):
                    all_interests[item.strip().title()] += 1
                elif isinstance(item, dict):
                    for sub in item.get("items", []):
                        all_interests[sub.strip().title()] += 1

            for hobby in interests.get("hobbies", []):
                if isinstance(hobby, str):
                    all_hobbies[hobby.strip().title()] += 1

            for value in interests.get("values", []):
                if isinstance(value, str):
                    all_values[value.strip().title()] += 1

        return {
            "interests": [
                {"category": k, "items": [k], "count": v}
                for k, v in all_interests.most_common(15)
            ],
            "hobbies": [h for h, _ in all_hobbies.most_common(10)],
            "values": [v for v, _ in all_values.most_common(10)],
        }

    # ── Keyword Aggregation ──────────────────────────────────────────────

    def _aggregate_keywords(self, analyses: list[dict]) -> dict:
        """Merge keywords and entities across all answers."""
        all_keywords: Counter = Counter()
        all_entities: list[dict] = []
        seen_entities: set = set()
        all_topics: Counter = Counter()

        for qa in analyses:
            kw = qa.get("keywords_entities", {})
            for keyword in kw.get("keywords", []):
                if isinstance(keyword, str):
                    all_keywords[keyword.lower()] += 1

            for entity in kw.get("entities", []):
                key = (entity.get("text", ""), entity.get("label", ""))
                if key not in seen_entities:
                    seen_entities.add(key)
                    all_entities.append(entity)

            for topic in kw.get("topics", []):
                if isinstance(topic, str):
                    all_topics[topic.lower()] += 1

        return {
            "keywords": [k for k, _ in all_keywords.most_common(20)],
            "entities": all_entities[:15],
            "topics": [t for t, _ in all_topics.most_common(10)],
        }

    # ── Consistency Score ────────────────────────────────────────────────

    def _compute_consistency(self, analyses: list[dict]) -> float:
        """
        Compute how consistent the user's responses are across questions.
        Low std-dev in sentiment/emotion = high consistency.
        """
        sentiment_scores = []
        for qa in analyses:
            sent = qa.get("sentiment", {})
            score = sent.get("overall_score", sent.get("score"))
            if score is not None and isinstance(score, (int, float)):
                sentiment_scores.append(score)

        if len(sentiment_scores) < 2:
            return 1.0  # Not enough data, assume consistent

        std = statistics.stdev(sentiment_scores)
        # Convert std to a 0-1 consistency score (lower std = higher consistency)
        # Using sigmoid-like mapping: consistency = 1 - min(std / 0.5, 1)
        consistency = max(0.0, 1.0 - min(std / 0.5, 1.0))
        return consistency

    # ── Empty Profile ────────────────────────────────────────────────────

    def _empty_profile(self, normalized: dict) -> dict:
        normalized_clean = dict(normalized)
        normalized_clean.pop("_meta", None)
        return {
            "audio_features": {},
            "emotion": {"primary_emotion": "Neutral", "confidence": 0.0},
            "sentiment": {"overall_label": "NEUTRAL", "overall_score": 0.0},
            "interests": {"interests": [], "hobbies": [], "values": []},
            "personality": {"big_five": {}, "communication_style": "Unknown"},
            "keywords_entities": {"keywords": [], "entities": [], "topics": []},
            "transcript": normalized_clean,
            "cross_confidence": 0.0,
            "aggregation_stats": {
                "emotion_distribution": {},
                "avg_confidence": 0.0,
                "avg_sentiment_score": 0.0,
                "personality_weighted": {},
                "consistency_score": 0.0,
                "total_questions_detected": 0,
                "total_questions_possible": 35,
            },
        }

    # ── Fallback ────────────────────────────────────────────────────────

    async def _fallback(self, input_data: dict[str, Any], error: Exception) -> dict[str, Any]:
        normalized = input_data.get("normalized_transcript", {})
        result = self._empty_profile(normalized)
        result["fallback_reason"] = str(error)
        return result
