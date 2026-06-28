"""
Audio Feature Extraction Service
- Pitch (f0 via pyin)
- Energy (RMS)
- Speaking rate (syllables/sec estimate)
- Hesitation count
- Pause duration stats
- Voice stability (pitch variance)
- Speaking style classification
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.services.base import BaseService

try:
    import librosa

    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


class AudioFeaturesService(BaseService):
    name = "audio_features"

    def __init__(self) -> None:
        super().__init__()
        if not HAS_LIBROSA:
            self._fallback_mode = True
            self.logger.warning("librosa not available – using rule-based fallback")

    # ── Core Processing ─────────────────────────────────────────────────

    async def _process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        if not HAS_LIBROSA:
            raise RuntimeError("librosa not installed")

        import io

        audio_bytes: bytes = input_data["processed_audio"]
        sr: int = input_data.get("sample_rate", 16000)

        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=sr, mono=True)
        duration = len(y) / sr

        # ── Pitch (f0) ──────────────────────────────────────────────────
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"),
            sr=sr
        )
        f0_valid = f0[~np.isnan(f0)] if f0 is not None else np.array([0.0])
        if len(f0_valid) == 0:
            f0_valid = np.array([0.0])

        pitch_mean = float(np.mean(f0_valid))
        pitch_std = float(np.std(f0_valid))
        pitch_min = float(np.min(f0_valid))
        pitch_max = float(np.max(f0_valid))

        # Voice stability: inverse of coefficient of variation (0–1 scale)
        cv = pitch_std / (pitch_mean + 1e-10)
        voice_stability = float(max(0.0, min(1.0, 1.0 - cv)))

        # ── Energy (RMS) ────────────────────────────────────────────────
        rms = librosa.feature.rms(y=y)[0]
        energy_mean = float(np.mean(rms))
        energy_std = float(np.std(rms))

        # ── Pauses & hesitations ────────────────────────────────────────
        intervals = librosa.effects.split(y, top_db=30)
        pause_durations = []
        for i in range(1, len(intervals)):
            gap_start = intervals[i - 1][1]
            gap_end = intervals[i][0]
            gap_sec = (gap_end - gap_start) / sr
            if gap_sec > 0.15:  # Only count pauses > 150ms
                pause_durations.append(gap_sec)

        pause_count = len(pause_durations)
        pause_duration_mean = float(np.mean(pause_durations)) if pause_durations else 0.0
        pause_duration_total = float(sum(pause_durations))

        # Hesitations: short pauses between 150ms–500ms
        hesitation_count = sum(1 for p in pause_durations if 0.15 <= p <= 0.5)

        # ── Speaking rate ───────────────────────────────────────────────
        # Estimate syllables from energy peaks
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        peaks = librosa.util.peak_pick(
            onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.3, wait=5
        )
        speech_duration = duration - pause_duration_total
        speaking_rate = len(peaks) / max(speech_duration, 0.1)

        # ── Style classification ────────────────────────────────────────
        speaking_style = self._classify_style(
            speaking_rate, pitch_std, pause_count, energy_mean, duration
        )

        return {
            "pitch_mean": round(pitch_mean, 2),
            "pitch_std": round(pitch_std, 2),
            "pitch_min": round(pitch_min, 2),
            "pitch_max": round(pitch_max, 2),
            "energy_mean": round(energy_mean, 6),
            "energy_std": round(energy_std, 6),
            "speaking_rate_sps": round(speaking_rate, 2),
            "pause_count": pause_count,
            "pause_duration_mean": round(pause_duration_mean, 3),
            "pause_duration_total": round(pause_duration_total, 3),
            "hesitation_count": hesitation_count,
            "voice_stability": round(voice_stability, 3),
            "speaking_style": speaking_style,
        }

    # ── Style Classification ────────────────────────────────────────────

    @staticmethod
    def _classify_style(
        rate: float, pitch_var: float, pauses: int, energy: float, duration: float
    ) -> str:
        if rate > 5.0 and pitch_var > 40:
            return "enthusiastic"
        elif rate > 4.0 and pitch_var > 25:
            return "animated"
        elif rate < 2.5 and pauses > 5:
            return "thoughtful"
        elif rate < 2.0:
            return "measured"
        elif energy < 0.01:
            return "soft-spoken"
        elif energy > 0.08:
            return "assertive"
        else:
            return "conversational"

    # ── Fallback ────────────────────────────────────────────────────────

    async def _fallback(self, input_data: dict[str, Any], error: Exception) -> dict[str, Any]:
        return {
            "pitch_mean": 165.0,
            "pitch_std": 30.0,
            "pitch_min": 100.0,
            "pitch_max": 250.0,
            "energy_mean": 0.04,
            "energy_std": 0.01,
            "speaking_rate_sps": 3.5,
            "pause_count": 4,
            "pause_duration_mean": 0.4,
            "pause_duration_total": 1.6,
            "hesitation_count": 2,
            "voice_stability": 0.72,
            "speaking_style": "conversational",
            "fallback_reason": str(error),
        }
