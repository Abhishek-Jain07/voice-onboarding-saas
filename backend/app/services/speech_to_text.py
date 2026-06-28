"""
Speech-to-Text Service
- faster-whisper (CTranslate2 backend)
- Word-level timestamps
- Language detection
- Confidence score
- Fallback to mock transcription
"""

from __future__ import annotations

import io
from typing import Any

from app.config import settings
from app.services.base import BaseService

try:
    from faster_whisper import WhisperModel

    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False


class SpeechToTextService(BaseService):
    name = "speech_to_text"

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        if not HAS_WHISPER:
            self._fallback_mode = True
            self.logger.warning("faster-whisper not installed – using mock STT")

    def _load_model(self) -> None:
        if self._model is not None:
            return
        if not HAS_WHISPER:
            raise RuntimeError("faster-whisper not installed")
        self.logger.info(
            "Loading Whisper model",
            extra={
                "service": self.name,
                "details": {
                    "size": settings.whisper_model_size,
                    "device": settings.whisper_device,
                    "compute_type": settings.whisper_compute_type,
                },
            },
        )
        self._model = WhisperModel(
            settings.whisper_model_size,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )

    # ── Core Processing ─────────────────────────────────────────────────

    async def _process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        self._load_model()
        audio_bytes: bytes = input_data["processed_audio"]
        sample_rate: int = input_data.get("sample_rate", 16000)

        # Write to temp buffer (faster-whisper accepts file-like objects)
        import tempfile, os

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            segments, info = self._model.transcribe(
                tmp_path,
                beam_size=5,
                word_timestamps=True,
                vad_filter=True,
            )

            words = []
            full_text_parts = []
            total_prob = 0.0
            word_count = 0

            for segment in segments:
                full_text_parts.append(segment.text.strip())
                if segment.words:
                    for w in segment.words:
                        words.append({
                            "word": w.word.strip(),
                            "start": round(w.start, 3),
                            "end": round(w.end, 3),
                            "probability": round(w.probability, 4),
                        })
                        total_prob += w.probability
                        word_count += 1

            full_text = " ".join(full_text_parts)
            avg_confidence = total_prob / word_count if word_count > 0 else 0.0

            return {
                "text": full_text,
                "language": info.language or "en",
                "language_probability": round(info.language_probability, 4),
                "confidence": round(avg_confidence, 4),
                "words": words,
                "duration_seconds": round(info.duration, 3),
            }
        finally:
            os.unlink(tmp_path)

    # ── Fallback ────────────────────────────────────────────────────────

    async def _fallback(self, input_data: dict[str, Any], error: Exception) -> dict[str, Any]:
        """Return a mock transcription result for testing."""
        duration = input_data.get("duration_seconds", 0.0)
        return {
            "text": (
                "Hello, my name is Alex. I love hiking, cooking Italian food, "
                "and watching sci-fi movies. I'm looking for someone who is kind, "
                "adventurous, and has a great sense of humor. I work in software "
                "engineering and enjoy traveling to new places on weekends."
            ),
            "language": "en",
            "language_probability": 0.95,
            "confidence": 0.85,
            "words": [],
            "duration_seconds": duration if duration else 15.0,
            "mock": True,
            "fallback_reason": str(error),
        }
