"""
Prompt Builder Service (v2 – Structured Profile Only)
─────────────────────────────────────────────────────
Build a compact, structured prompt from aggregated profile data.
NO raw transcript is sent to the LLM — only structured, aggregated data.
Target: < 500 tokens (>90% reduction from v1).
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

        # ── Aggregation Statistics ──────────────────────────────────
        stats = profile.get("aggregation_stats", {})
        if stats:
            detected = stats.get("total_questions_detected", 0)
            consistency = stats.get("consistency_score", 0)
            parts.append(
                f"[Analysis] {detected} questions analyzed, "
                f"consistency={consistency:.2f}"
            )

        # ── Emotion Distribution ────────────────────────────────────
        emotion = profile.get("emotion", {})
        distribution = emotion.get("distribution", {})
        if distribution:
            dist_str = ", ".join(
                f"{k}:{v:.0%}" for k, v in list(distribution.items())[:4]
            )
            parts.append(f"[EmotionDistribution] {dist_str}")
        elif emotion.get("primary_emotion"):
            parts.append(
                f"[Emotion] primary={emotion['primary_emotion']}, "
                f"secondary={emotion.get('secondary_emotion', 'n/a')}"
            )

        # ── Sentiment ───────────────────────────────────────────────
        sentiment = profile.get("sentiment", {})
        if sentiment.get("overall_label"):
            parts.append(
                f"[Sentiment] {sentiment['overall_label']} "
                f"(score={sentiment.get('overall_score', 0):.2f}, "
                f"std={sentiment.get('std_dev', 0):.2f})"
            )

        # ── Personality (Weighted Big Five) ─────────────────────────
        personality = profile.get("personality", {})
        big_five = personality.get("big_five", {})
        if big_five:
            traits = []
            for trait, score in big_five.items():
                if isinstance(score, (int, float)):
                    level = "high" if score > 0.65 else ("low" if score < 0.35 else "moderate")
                    traits.append(f"{trait}={level}({score:.2f})")
            parts.append(f"[Big5] {', '.join(traits)}")

        comm_style = personality.get("communication_style", "")
        if comm_style:
            parts.append(f"[CommStyle] {comm_style}")

        # ── Interests (Aggregated & Ranked) ─────────────────────────
        interests = profile.get("interests", {})
        interest_list = interests.get("interests", [])
        if interest_list:
            interest_strs = []
            for cat in interest_list[:8]:
                if isinstance(cat, dict):
                    name = cat.get("category", "")
                    count = cat.get("count", 1)
                    interest_strs.append(f"{name}(x{count})")
                elif isinstance(cat, str):
                    interest_strs.append(cat)
            parts.append(f"[Interests] {', '.join(interest_strs)}")

        hobbies = interests.get("hobbies", [])
        if hobbies:
            parts.append(f"[Hobbies] {', '.join(hobbies[:6])}")

        values = interests.get("values", [])
        if values:
            parts.append(f"[Values] {', '.join(values[:5])}")

        # ── Audio Features (Compact) ────────────────────────────────
        audio = profile.get("audio_features", {})
        if audio:
            parts.append(
                f"[Voice] style={audio.get('speaking_style', 'n/a')}, "
                f"rate={audio.get('speaking_rate_sps', 0):.1f}sps, "
                f"stability={audio.get('voice_stability', 0):.2f}"
            )

        # ── Keywords ────────────────────────────────────────────────
        kw = profile.get("keywords_entities", {})
        keywords = kw.get("keywords", [])
        if keywords:
            parts.append(f"[Keywords] {', '.join(keywords[:10])}")

        entities = kw.get("entities", [])
        if entities:
            ent_strs = [f"{e['text']}({e['label']})" for e in entities[:5]
                       if isinstance(e, dict)]
            if ent_strs:
                parts.append(f"[Entities] {', '.join(ent_strs)}")

        # ── Conversation Summary ────────────────────────────────────
        summary_data = profile.get("conversation_summary", {})
        if summary_data.get("summary"):
            parts.append(f"[Summary] {summary_data['summary'][:200]}")

        prefs = summary_data.get("stated_preferences", [])
        if prefs:
            parts.append(f"[Preferences] {', '.join(prefs[:5])}")

        goals = summary_data.get("relationship_goals", [])
        if goals:
            parts.append(f"[Goals] {', '.join(goals[:3])}")

        # ── Confidence ──────────────────────────────────────────────
        cross_conf = profile.get("cross_confidence", 0.0)
        parts.append(f"[Confidence] {cross_conf:.2f}")

        parts.append("")
        parts.append(
            "Write engaging, warm, authentic content. "
            "Be specific based on the data above. "
            "Use the emotion distribution and personality traits to make the profile feel genuine."
        )

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
