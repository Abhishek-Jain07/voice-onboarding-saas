"""
Memory Service
- Versioned profile storage
- Confidence tracking
- Incremental updates
- JSON file-based persistence
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import settings
from app.services.base import BaseService


class MemoryService(BaseService):
    name = "memory"

    # ── Core Processing ─────────────────────────────────────────────────

    async def _process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        session_id: str = input_data.get("session_id") or str(uuid.uuid4())
        profile: dict = input_data.get("aggregated_profile", {})
        confidence: float = profile.get("cross_confidence", 0.0)

        # Load existing memory
        memory = self._load_memory(session_id)

        # Create new version
        new_version = {
            "version": memory["current_version"] + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "profile": profile,
            "confidence": confidence,
        }

        # If prior versions exist, merge with confidence weighting
        if memory["versions"]:
            latest = memory["versions"][-1]
            merged_profile = self._merge_profiles(
                latest.get("profile", {}),
                profile,
                latest.get("confidence", 0.0),
                confidence,
            )
            new_version["profile"] = merged_profile
            new_version["confidence"] = max(
                latest.get("confidence", 0.0),
                confidence,
            )

        memory["versions"].append(new_version)
        memory["current_version"] = new_version["version"]

        # Save
        self._save_memory(session_id, memory)

        return {
            "session_id": session_id,
            "version": new_version["version"],
            "total_versions": len(memory["versions"]),
            "profile": new_version["profile"],
            "confidence": new_version["confidence"],
        }

    def _merge_profiles(
        self,
        old_profile: dict,
        new_profile: dict,
        old_conf: float,
        new_conf: float,
    ) -> dict:
        """Merge two profiles with confidence-weighted priority."""
        merged = {}

        all_keys = set(list(old_profile.keys()) + list(new_profile.keys()))

        for key in all_keys:
            old_val = old_profile.get(key)
            new_val = new_profile.get(key)

            if new_val is None:
                merged[key] = old_val
            elif old_val is None:
                merged[key] = new_val
            elif isinstance(new_val, dict) and isinstance(old_val, dict):
                # Recursive merge for nested dicts
                merged[key] = self._merge_profiles(old_val, new_val, old_conf, new_conf)
            elif isinstance(new_val, list) and isinstance(old_val, list):
                # Union of lists (deduplicated)
                seen = set()
                combined = []
                for item in old_val + new_val:
                    item_key = json.dumps(item, sort_keys=True, default=str) if isinstance(item, dict) else str(item)
                    if item_key not in seen:
                        seen.add(item_key)
                        combined.append(item)
                merged[key] = combined
            else:
                # Take higher-confidence value
                merged[key] = new_val if new_conf >= old_conf else old_val

        return merged

    # ── Persistence ─────────────────────────────────────────────────────

    def _get_path(self, session_id: str) -> Path:
        return settings.profile_path / f"{session_id}.json"

    def _load_memory(self, session_id: str) -> dict:
        path = self._get_path(session_id)
        if path.exists():
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {
            "session_id": session_id,
            "versions": [],
            "current_version": 0,
        }

    def _save_memory(self, session_id: str, memory: dict) -> None:
        path = self._get_path(session_id)
        try:
            with open(path, "w") as f:
                json.dump(memory, f, indent=2, default=str)
            self.logger.info(
                "Profile saved",
                extra={
                    "service": self.name,
                    "details": {"session_id": session_id, "path": str(path)},
                },
            )
        except IOError as e:
            self.logger.error(f"Failed to save profile: {e}", extra={"service": self.name})

    # ── Fallback ────────────────────────────────────────────────────────

    async def _fallback(self, input_data: dict[str, Any], error: Exception) -> dict[str, Any]:
        session_id = input_data.get("session_id") or str(uuid.uuid4())
        profile = input_data.get("aggregated_profile", {})
        return {
            "session_id": session_id,
            "version": 1,
            "total_versions": 1,
            "profile": profile,
            "confidence": profile.get("cross_confidence", 0.0),
            "fallback_reason": str(error),
        }
