"""
Emotion Classification Service
- Audio-based emotion detection using pitch, energy, tempo patterns
- Rule-based approach with optional lightweight model
- Returns primary emotion, secondary emotion, confidence
"""

from __future__ import annotations

from typing import Any

from app.services.base import BaseService


class EmotionService(BaseService):
    name = "emotion"

    # ── Core Processing ─────────────────────────────────────────────────

    async def _process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        Rule-based emotion classification from audio features.
        Uses pitch, energy, speaking rate, and voice stability.
        """
        # Extract audio features (passed from audio_features service or input)
        pitch_mean = input_data.get("pitch_mean", 165.0)
        pitch_std = input_data.get("pitch_std", 30.0)
        energy_mean = input_data.get("energy_mean", 0.04)
        speaking_rate = input_data.get("speaking_rate_sps", 3.5)
        voice_stability = input_data.get("voice_stability", 0.7)
        pause_count = input_data.get("pause_count", 3)

        # Compute emotion scores using rule-based heuristics
        scores = self._compute_emotion_scores(
            pitch_mean, pitch_std, energy_mean,
            speaking_rate, voice_stability, pause_count
        )

        # Sort by score
        sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary = sorted_emotions[0]
        secondary = sorted_emotions[1] if len(sorted_emotions) > 1 else None

        return {
            "primary_emotion": primary[0],
            "secondary_emotion": secondary[0] if secondary else None,
            "confidence": round(primary[1], 3),
            "emotion_scores": {k: round(v, 3) for k, v in scores.items()},
        }

    @staticmethod
    def _compute_emotion_scores(
        pitch_mean: float,
        pitch_std: float,
        energy_mean: float,
        speaking_rate: float,
        voice_stability: float,
        pause_count: int,
    ) -> dict[str, float]:
        """
        Heuristic emotion scoring based on prosodic features.

        Research-based associations:
        - Happy/Excited: high pitch, high energy, fast rate, high variability
        - Sad: low pitch, low energy, slow rate, high stability
        - Angry: high pitch, very high energy, fast rate, low stability
        - Fearful: high pitch, medium energy, fast rate, low stability, many pauses
        - Calm: medium pitch, low energy, slow rate, high stability
        - Neutral: moderate everything
        """
        scores: dict[str, float] = {}

        # Happy: high pitch + high energy + fast
        happy_score = 0.0
        if pitch_mean > 180:
            happy_score += 0.25
        if pitch_std > 35:
            happy_score += 0.15
        if energy_mean > 0.05:
            happy_score += 0.2
        if speaking_rate > 4.0:
            happy_score += 0.2
        if voice_stability > 0.5:
            happy_score += 0.1
        scores["happy"] = min(happy_score + 0.1, 1.0)

        # Excited: very high pitch + very high energy + very fast
        excited_score = 0.0
        if pitch_mean > 200:
            excited_score += 0.3
        if pitch_std > 50:
            excited_score += 0.2
        if energy_mean > 0.07:
            excited_score += 0.25
        if speaking_rate > 5.0:
            excited_score += 0.2
        scores["excited"] = min(excited_score + 0.05, 1.0)

        # Sad: low pitch + low energy + slow
        sad_score = 0.0
        if pitch_mean < 140:
            sad_score += 0.3
        if pitch_std < 20:
            sad_score += 0.15
        if energy_mean < 0.02:
            sad_score += 0.25
        if speaking_rate < 2.5:
            sad_score += 0.2
        scores["sad"] = min(sad_score + 0.05, 1.0)

        # Angry: high energy + fast + unstable
        angry_score = 0.0
        if energy_mean > 0.08:
            angry_score += 0.3
        if speaking_rate > 4.5:
            angry_score += 0.2
        if voice_stability < 0.4:
            angry_score += 0.25
        if pitch_std > 45:
            angry_score += 0.15
        scores["angry"] = min(angry_score + 0.05, 1.0)

        # Fearful: high pitch + many pauses + unstable
        fearful_score = 0.0
        if pitch_mean > 190:
            fearful_score += 0.2
        if pause_count > 8:
            fearful_score += 0.25
        if voice_stability < 0.4:
            fearful_score += 0.25
        if speaking_rate > 4.0:
            fearful_score += 0.15
        scores["fearful"] = min(fearful_score + 0.05, 1.0)

        # Calm: medium pitch + low energy + stable
        calm_score = 0.0
        if 130 <= pitch_mean <= 180:
            calm_score += 0.2
        if energy_mean < 0.04:
            calm_score += 0.2
        if voice_stability > 0.7:
            calm_score += 0.25
        if speaking_rate < 3.5:
            calm_score += 0.2
        scores["calm"] = min(calm_score + 0.1, 1.0)

        # Neutral: everything moderate
        neutral_score = 0.0
        if 140 <= pitch_mean <= 200:
            neutral_score += 0.2
        if 0.02 <= energy_mean <= 0.06:
            neutral_score += 0.2
        if 2.5 <= speaking_rate <= 4.5:
            neutral_score += 0.2
        if 0.4 <= voice_stability <= 0.8:
            neutral_score += 0.2
        scores["neutral"] = min(neutral_score + 0.15, 1.0)

        # Surprised: high pitch jump + fast
        surprised_score = 0.0
        if pitch_std > 60:
            surprised_score += 0.35
        if energy_mean > 0.06:
            surprised_score += 0.2
        if speaking_rate > 4.5:
            surprised_score += 0.15
        scores["surprised"] = min(surprised_score + 0.05, 1.0)

        # Normalize so they sum to ~1
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}

        return scores

    # ── Fallback ────────────────────────────────────────────────────────

    async def _fallback(self, input_data: dict[str, Any], error: Exception) -> dict[str, Any]:
        return {
            "primary_emotion": "neutral",
            "secondary_emotion": "calm",
            "confidence": 0.5,
            "emotion_scores": {
                "neutral": 0.35,
                "calm": 0.25,
                "happy": 0.15,
                "sad": 0.08,
                "excited": 0.07,
                "angry": 0.04,
                "fearful": 0.03,
                "surprised": 0.03,
            },
            "fallback_reason": str(error),
        }
