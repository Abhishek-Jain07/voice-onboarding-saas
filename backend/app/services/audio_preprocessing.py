"""
Audio Preprocessing Service
- VAD (energy-based)
- Noise reduction (spectral gating)
- Audio normalization
- Sample rate conversion to 16 kHz
- Silence removal
- Format conversion via pydub/ffmpeg
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from app.services.base import BaseService

# Optional heavy imports with fallback
try:
    import librosa
    import soundfile as sf

    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    from pydub import AudioSegment

    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False

TARGET_SR = 16_000
SUPPORTED_FORMATS = {"wav", "mp3", "webm", "ogg", "m4a", "flac", "mp4", "wma"}


class AudioPreprocessingService(BaseService):
    name = "audio_preprocessing"

    def __init__(self) -> None:
        super().__init__()
        if not HAS_LIBROSA:
            self._fallback_mode = True
            self.logger.warning("librosa not available – running in fallback mode")

    # ── Core Processing ─────────────────────────────────────────────────

    async def _process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        audio_bytes: bytes = input_data["audio_bytes"]
        filename: str = input_data.get("filename", "audio.wav")
        ext = Path(filename).suffix.lstrip(".").lower()
        if ext not in SUPPORTED_FORMATS:
            ext = "wav"

        # Step 1 – Convert to WAV (16-bit PCM) via pydub if not already wav
        wav_bytes = self._to_wav(audio_bytes, ext)

        # Step 2 – Load with librosa
        if not HAS_LIBROSA:
            raise RuntimeError("librosa not installed")

        y, sr = librosa.load(io.BytesIO(wav_bytes), sr=TARGET_SR, mono=True)

        # Step 3 – Normalize amplitude
        y = self._normalize(y)

        # Step 4 – Simple spectral noise gate
        y = self._reduce_noise(y, sr)

        # Step 5 – Remove silence (energy-based VAD)
        y_trimmed, _ = librosa.effects.trim(y, top_db=25)

        # Step 6 – Remove internal silence
        intervals = librosa.effects.split(y_trimmed, top_db=30)
        if len(intervals) > 0:
            chunks = [y_trimmed[s:e] for s, e in intervals]
            # Insert tiny silence between chunks to avoid harsh joins
            silence = np.zeros(int(0.05 * sr), dtype=np.float32)
            parts: list[np.ndarray] = []
            for i, chunk in enumerate(chunks):
                parts.append(chunk)
                if i < len(chunks) - 1:
                    parts.append(silence)
            y_clean = np.concatenate(parts)
        else:
            y_clean = y_trimmed

        # Step 7 – Write out as 16-bit WAV
        buf = io.BytesIO()
        sf.write(buf, y_clean, sr, format="WAV", subtype="PCM_16")
        processed_bytes = buf.getvalue()

        # Compute audio info
        rms = float(np.sqrt(np.mean(y_clean**2)))
        rms_db = float(20 * np.log10(rms + 1e-10))

        return {
            "processed_audio": processed_bytes,
            "sample_rate": sr,
            "duration_seconds": round(len(y_clean) / sr, 3),
            "channels": 1,
            "format": "wav",
            "rms_db": round(rms_db, 2),
            "original_duration": round(len(y) / sr, 3),
        }

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _to_wav(audio_bytes: bytes, ext: str) -> bytes:
        """Convert any supported format to WAV bytes via pydub."""
        if ext == "wav":
            return audio_bytes
        if not HAS_PYDUB:
            # Try to pass through anyway – librosa may handle it
            return audio_bytes
        segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format=ext)
        segment = segment.set_frame_rate(TARGET_SR).set_channels(1).set_sample_width(2)
        buf = io.BytesIO()
        segment.export(buf, format="wav")
        return buf.getvalue()

    @staticmethod
    def _normalize(y: np.ndarray) -> np.ndarray:
        """Peak-normalise to -1..+1."""
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak * 0.95
        return y

    @staticmethod
    def _reduce_noise(y: np.ndarray, sr: int) -> np.ndarray:
        """Simple spectral gating noise reduction."""
        try:
            # Use first 0.5 s as noise profile
            noise_len = min(int(0.5 * sr), len(y) // 4)
            noise_sample = y[:noise_len]
            noise_stft = np.abs(librosa.stft(noise_sample))
            noise_profile = np.mean(noise_stft, axis=1, keepdims=True)

            stft = librosa.stft(y)
            mag = np.abs(stft)
            phase = np.angle(stft)

            # Gate: subtract noise floor, clamp
            mag_clean = np.maximum(mag - 1.5 * noise_profile, 0.0)
            stft_clean = mag_clean * np.exp(1j * phase)
            y_clean = librosa.istft(stft_clean, length=len(y))
            return y_clean
        except Exception:
            return y  # If noise reduction fails, return original

    # ── Fallback ────────────────────────────────────────────────────────

    async def _fallback(self, input_data: dict[str, Any], error: Exception) -> dict[str, Any]:
        """Return raw audio with minimal metadata."""
        audio_bytes: bytes = input_data["audio_bytes"]
        return {
            "processed_audio": audio_bytes,
            "sample_rate": TARGET_SR,
            "duration_seconds": 0.0,
            "channels": 1,
            "format": "wav",
            "rms_db": 0.0,
            "original_duration": 0.0,
            "fallback_reason": str(error),
        }
