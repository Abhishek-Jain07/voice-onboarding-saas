"""
Prompt Builder Service
- Build compact, structured prompt from aggregated profile
- Target: < 500 tokens
- Includes all extracted features in structured format
"""

from __future__ import annotations

from typing import Any

from app.services.base import BaseService


class PromptBuilderService(BaseService):
    name = "prompt_builder"

    # ── Core Processing ─────────────────────────────────────────────────

    async def _process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        profile: dict = input_data.get("profile", {})

        prompt = self._build_prompt(profile)

        # Rough token count
        token_count = len(prompt) // 4

        return {
            "prompt": prompt,
            "token_count": token_count,
        }

    def _build_prompt(self, profile: dict) -> str:
        """Build a compact, structured prompt under 500 tokens."""
        parts = [
            "Generate a dating profile from this voice analysis data.",
            "Return JSON with keys: personality_summary, dating_bio, compatibility_features (list), "
            "ice_breakers (list), green_flags (list), red_flags (list), conversation_style, "
            "match_recommendations (list).",
            "",
            "=== VOICE ANALYSIS DATA ===",
        ]

        # Transcript summary
        summary_data = profile.get("conversation_summary", {})
        if summary_data.get("summary"):
            parts.append(f"[Summary] {summary_data['summary'][:200]}")

        # Stated preferences
        prefs = summary_data.get("stated_preferences", [])
        if prefs:
            parts.append(f"[Preferences] {', '.join(prefs[:5])}")

        # Relationship goals
        goals = summary_data.get("relationship_goals", [])
        if goals:
            parts.append(f"[Goals] {', '.join(goals[:3])}")

        # Interests
        interests = profile.get("interests", {})
        interest_list = interests.get("interests", [])
        if interest_list:
            interest_strs = []
            for cat in interest_list[:5]:
                items = cat.get("items", [])
                if items:
                    interest_strs.append(f"{cat['category']}: {', '.join(items[:3])}")
                else:
                    interest_strs.append(cat["category"])
            parts.append(f"[Interests] {'; '.join(interest_strs)}")

        # Personality
        personality = profile.get("personality", {})
        big_five = personality.get("big_five", {})
        if big_five:
            traits = []
            for trait, score in big_five.items():
                if isinstance(score, (int, float)):
                    level = "high" if score > 0.65 else ("low" if score < 0.35 else "moderate")
                    traits.append(f"{trait}={level}")
            parts.append(f"[Big5] {', '.join(traits)}")

        comm_style = personality.get("communication_style", "")
        if comm_style:
            parts.append(f"[CommStyle] {comm_style}")

        attach_style = personality.get("attachment_style", "")
        if attach_style:
            parts.append(f"[Attachment] {attach_style}")

        # Emotion & Sentiment
        emotion = profile.get("emotion", {})
        if emotion.get("primary_emotion"):
            parts.append(
                f"[Emotion] primary={emotion['primary_emotion']}, "
                f"secondary={emotion.get('secondary_emotion', 'n/a')}"
            )

        sentiment = profile.get("sentiment", {})
        if sentiment.get("overall_label"):
            parts.append(
                f"[Sentiment] {sentiment['overall_label']} "
                f"(score={sentiment.get('overall_score', 0):.2f})"
            )

        # Audio features (compact)
        audio = profile.get("audio_features", {})
        if audio:
            parts.append(
                f"[Voice] style={audio.get('speaking_style', 'n/a')}, "
                f"rate={audio.get('speaking_rate_sps', 0):.1f}sps, "
                f"stability={audio.get('voice_stability', 0):.2f}"
            )

        # Keywords
        kw = profile.get("keywords_entities", {})
        keywords = kw.get("keywords", [])
        if keywords:
            parts.append(f"[Keywords] {', '.join(keywords[:8])}")

        # Entities
        entities = kw.get("entities", [])
        if entities:
            ent_strs = [f"{e['text']}({e['label']})" for e in entities[:5]]
            parts.append(f"[Entities] {', '.join(ent_strs)}")

        # Confidence
        cross_conf = profile.get("cross_confidence", 0.0)
        parts.append(f"[Confidence] {cross_conf:.2f}")

        parts.append("")
        parts.append("Write engaging, warm, authentic content. Be specific based on the data above.")

        return "\n".join(parts)

    # ── Fallback ────────────────────────────────────────────────────────

    async def _fallback(self, input_data: dict[str, Any], error: Exception) -> dict[str, Any]:
        return {
            "prompt": (
                "Generate a dating profile. The user seems friendly and adventurous. "
                "Return JSON with keys: personality_summary, dating_bio, "
                "compatibility_features, ice_breakers, green_flags, red_flags, "
                "conversation_style, match_recommendations."
            ),
            "token_count": 50,
            "fallback_reason": str(error),
        }
