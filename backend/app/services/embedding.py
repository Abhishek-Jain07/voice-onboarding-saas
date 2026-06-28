"""
Embedding Service (Optional)
- Sentence embeddings for semantic similarity
- Uses sentence-transformers when available
- Fallback returns empty embeddings
"""

from __future__ import annotations

from typing import Any

from app.services.base import BaseService

try:
    from sentence_transformers import SentenceTransformer

    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False


class EmbeddingService(BaseService):
    name = "embedding"

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        if not HAS_SBERT:
            self._fallback_mode = True
            self.logger.warning("sentence-transformers not available – embeddings disabled")

    def _load_model(self) -> None:
        if self._model is not None:
            return
        if not HAS_SBERT:
            raise RuntimeError("sentence-transformers not installed")
        self.logger.info("Loading sentence-transformer model", extra={"service": self.name})
        self._model = SentenceTransformer("all-MiniLM-L6-v2")

    # ── Core Processing ─────────────────────────────────────────────────

    async def _process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        self._load_model()
        text: str = input_data.get("normalized_text", "") or input_data.get("text", "")
        sentences: list[str] = input_data.get("sentences", [text])

        if not sentences:
            return {"embeddings": [], "dimension": 0}

        embeddings = self._model.encode(sentences).tolist()

        return {
            "embeddings": embeddings,
            "dimension": len(embeddings[0]) if embeddings else 0,
            "num_sentences": len(sentences),
        }

    # ── Fallback ────────────────────────────────────────────────────────

    async def _fallback(self, input_data: dict[str, Any], error: Exception) -> dict[str, Any]:
        return {
            "embeddings": [],
            "dimension": 0,
            "num_sentences": 0,
            "fallback_reason": str(error),
        }
