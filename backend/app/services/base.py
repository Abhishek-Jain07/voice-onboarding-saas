"""
Abstract base service for all pipeline services.
Every service MUST inherit from BaseService.
"""

from __future__ import annotations

import time
import traceback
from abc import ABC, abstractmethod
from typing import Any

from app.logging_config import get_logger


class BaseService(ABC):
    """
    Base class for every AI pipeline service.

    Subclasses must implement:
        - name (str)
        - _process(input_data) -> dict
        - _fallback(input_data, error) -> dict   [optional but recommended]
    """

    name: str = "base"
    _fallback_mode: bool = False

    def __init__(self) -> None:
        self.logger = get_logger(f"service.{self.name}")
        self._initialised = False

    # ── Public API ───────────────────────────────────────────────────────

    async def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Run the service with timing + automatic fallback on error."""
        start = time.perf_counter()
        
        if getattr(self, "_fallback_mode", False):
            # Intentional fallback bypasses error tracking
            result = await self._fallback(input_data, Exception("Intentional fallback mode"))
            elapsed = (time.perf_counter() - start) * 1000
            self.logger.info(
                "Service running in fallback mode",
                extra={"service": self.name, "duration_ms": round(elapsed, 2)},
            )
            result["_meta"] = {
                "service": self.name,
                "duration_ms": round(elapsed, 2),
                "fallback": True,
            }
            return result

        try:
            result = await self._process(input_data)
            elapsed = (time.perf_counter() - start) * 1000
            self.logger.info(
                "Service completed",
                extra={"service": self.name, "duration_ms": round(elapsed, 2)},
            )
            result["_meta"] = {
                "service": self.name,
                "duration_ms": round(elapsed, 2),
                "fallback": False,
            }
            return result
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            self.logger.warning(
                "Service failed – using fallback",
                extra={
                    "service": self.name,
                    "duration_ms": round(elapsed, 2),
                    "error": str(exc),
                },
            )
            try:
                result = await self._fallback(input_data, exc)
            except Exception:
                result = {
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            result["_meta"] = {
                "service": self.name,
                "duration_ms": round(elapsed, 2),
                "fallback": True,
                "error": str(exc),
            }
            return result

    async def health_check(self) -> dict[str, Any]:
        """Return health status of this service."""
        return {
            "service": self.name,
            "status": "healthy",
            "fallback_mode": self._fallback_mode,
        }

    # ── Subclass hooks ───────────────────────────────────────────────────

    @abstractmethod
    async def _process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Core processing logic – MUST be implemented by subclasses."""
        raise NotImplementedError

    async def _fallback(self, input_data: dict[str, Any], error: Exception) -> dict[str, Any]:
        """Fallback logic when _process fails. Override for richer fallbacks."""
        return {"error": str(error), "fallback": True}
