"""
Dating Profile Generator Service
- Parse LLM response into structured sections
- Validate and enrich profile
- Handle malformed LLM output gracefully
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.services.base import BaseService


class DatingProfileService(BaseService):
    name = "dating_profile"

    EXPECTED_KEYS = {
        "personality_summary",
        "dating_bio",
        "compatibility_features",
        "ice_breakers",
        "green_flags",
        "red_flags",
        "conversation_style",
        "match_recommendations",
    }

    # ── Core Processing ─────────────────────────────────────────────────

    async def _process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        llm_response: str = input_data.get("response", "")
        profile_data: dict = input_data.get("profile", {})

        # Parse LLM response
        parsed = self._parse_response(llm_response)

        # Validate and fill defaults
        profile = self._validate_profile(parsed, profile_data)

        return profile

    def _parse_response(self, response: str) -> dict[str, Any]:
        """Try to parse JSON from LLM response, handling edge cases."""
        if not response:
            return {}

        # Try direct JSON parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find JSON object in the text
        brace_match = re.search(r"\{[\s\S]*\}", response)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        # Last resort: try to extract key-value pairs
        self.logger.warning("Could not parse LLM response as JSON", extra={"service": self.name})
        return {"raw_response": response}

    def _validate_profile(self, parsed: dict, profile_data: dict) -> dict[str, Any]:
        """Ensure all expected keys exist with proper types."""
        result = {}

        # String fields
        for key in ["personality_summary", "dating_bio", "conversation_style"]:
            val = parsed.get(key, "")
            result[key] = val if isinstance(val, str) else str(val) if val else ""

        # List fields
        for key in ["compatibility_features", "ice_breakers", "green_flags",
                     "red_flags", "match_recommendations"]:
            val = parsed.get(key, [])
            if isinstance(val, list):
                result[key] = [str(item) for item in val]
            elif isinstance(val, str):
                result[key] = [item.strip() for item in val.split(",") if item.strip()]
            else:
                result[key] = []

        # Enrich with extracted data if LLM output was sparse
        if not result["personality_summary"]:
            personality = profile_data.get("personality", {})
            style = personality.get("communication_style", "balanced")
            result["personality_summary"] = (
                f"Shows a {style} communication style with "
                f"a {personality.get('attachment_style', 'secure')} attachment pattern."
            )

        if not result["dating_bio"]:
            summary = profile_data.get("conversation_summary", {})
            prefs = summary.get("stated_preferences", [])
            result["dating_bio"] = (
                f"Looking for meaningful connections. "
                f"Interests include: {', '.join(prefs[:3]) if prefs else 'various activities'}."
            )

        return result

    # ── Fallback ────────────────────────────────────────────────────────

    async def _fallback(self, input_data: dict[str, Any], error: Exception) -> dict[str, Any]:
        profile_data = input_data.get("profile", {})
        personality = profile_data.get("personality", {})
        interests = profile_data.get("interests", {})
        summary = profile_data.get("conversation_summary", {})

        interest_items = []
        for cat in interests.get("interests", []):
            interest_items.extend(cat.get("items", []))

        prefs = summary.get("stated_preferences", [])
        goals = summary.get("relationship_goals", [])

        return {
            "personality_summary": (
                f"A {personality.get('communication_style', 'balanced')} communicator "
                f"with {personality.get('attachment_style', 'secure')} attachment style."
            ),
            "dating_bio": (
                f"Enjoys {', '.join(interest_items[:3]) if interest_items else 'various activities'}. "
                f"{'Preferences: ' + ', '.join(prefs[:3]) + '.' if prefs else ''}"
            ),
            "compatibility_features": prefs[:4] if prefs else ["Open to new experiences"],
            "ice_breakers": [
                "What's the best adventure you've been on?",
                "What are you most passionate about?",
                "If you could have dinner with anyone, who would it be?",
            ],
            "green_flags": [
                "Expressive communicator",
                "Diverse interests",
                "Self-aware",
            ],
            "red_flags": [],
            "conversation_style": personality.get("communication_style", "balanced"),
            "match_recommendations": goals[:4] if goals else [
                "Someone kind and adventurous",
            ],
            "fallback_reason": str(error),
        }
