"""
Conversation Summary Service
- Compress transcript to structured summary
- Key topics, stated preferences, relationship goals
- Token-efficient (max ~100 tokens)
"""

from __future__ import annotations

import re
from typing import Any

from app.services.base import BaseService


class ConversationSummaryService(BaseService):
    name = "conversation_summary"

    # ── Core Processing ─────────────────────────────────────────────────

    async def _process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        text: str = input_data.get("normalized_text", "") or input_data.get("text", "")
        sentences: list[str] = input_data.get("sentences", [])

        if not sentences:
            sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

        if not sentences:
            return {
                "summary": "",
                "key_topics": [],
                "stated_preferences": [],
                "relationship_goals": [],
                "token_count": 0,
            }

        # Extract key topics from the text
        key_topics = self._extract_topics(sentences)

        # Extract stated preferences
        preferences = self._extract_preferences(sentences)

        # Extract relationship goals
        goals = self._extract_relationship_goals(sentences)

        # Build compact summary
        summary_parts = []

        if key_topics:
            summary_parts.append(f"Topics: {', '.join(key_topics[:5])}")
        if preferences:
            summary_parts.append(f"Likes: {', '.join(preferences[:5])}")
        if goals:
            summary_parts.append(f"Goals: {', '.join(goals[:3])}")

        # Add condensed transcript highlights
        important_sentences = self._select_important_sentences(sentences, max_sentences=3)
        if important_sentences:
            summary_parts.append(f"Key quotes: {' | '.join(important_sentences)}")

        summary = ". ".join(summary_parts)

        # Rough token count (~4 chars per token)
        token_count = len(summary) // 4

        return {
            "summary": summary,
            "key_topics": key_topics,
            "stated_preferences": preferences,
            "relationship_goals": goals,
            "token_count": token_count,
        }

    # ── Extractors ──────────────────────────────────────────────────────

    @staticmethod
    def _extract_topics(sentences: list[str]) -> list[str]:
        topic_indicators = {
            "work": ["work", "job", "career", "office", "company", "profession"],
            "hobbies": ["hobby", "hobbies", "enjoy", "love doing", "passion", "free time"],
            "family": ["family", "parents", "siblings", "kids", "children", "mom", "dad"],
            "travel": ["travel", "trip", "visited", "vacation", "explore", "adventure"],
            "food": ["food", "cook", "eat", "restaurant", "cuisine", "recipe"],
            "fitness": ["gym", "workout", "exercise", "run", "yoga", "fitness"],
            "entertainment": ["movie", "show", "music", "book", "game", "read", "watch"],
            "relationships": ["dating", "relationship", "partner", "love", "looking for"],
            "education": ["study", "college", "university", "degree", "learn"],
            "pets": ["dog", "cat", "pet", "puppy", "kitten"],
        }

        text_lower = " ".join(sentences).lower()
        topics = []
        for topic, keywords in topic_indicators.items():
            if any(kw in text_lower for kw in keywords):
                topics.append(topic)

        return topics[:8]

    @staticmethod
    def _extract_preferences(sentences: list[str]) -> list[str]:
        preferences = []
        preference_patterns = [
            r"(?:i )?(?:love|enjoy|like|prefer|adore)\s+(.+?)(?:\.|,|$)",
            r"(?:my )?(?:favorite|favourite)\s+(?:\w+\s+)?(?:is|are)\s+(.+?)(?:\.|,|$)",
            r"i(?:'m| am) (?:into|passionate about|a fan of)\s+(.+?)(?:\.|,|$)",
        ]

        for sent in sentences:
            for pattern in preference_patterns:
                matches = re.findall(pattern, sent, re.IGNORECASE)
                for match in matches:
                    pref = match.strip()
                    if 2 < len(pref) < 60:
                        preferences.append(pref)

        return list(dict.fromkeys(preferences))[:8]  # Deduplicate, keep order

    @staticmethod
    def _extract_relationship_goals(sentences: list[str]) -> list[str]:
        goals = []
        goal_patterns = [
            r"(?:i(?:'m| am) )?looking for\s+(.+?)(?:\.|,|$)",
            r"i want (?:to find |a )?(.+?)(?:\.|,|$)",
            r"(?:my )?ideal (?:partner|match|person)\s+(?:is|would be)\s+(.+?)(?:\.|,|$)",
            r"i(?:'m| am) seeking\s+(.+?)(?:\.|,|$)",
            r"someone who (?:is |can )?(.+?)(?:\.|,|$)",
        ]

        for sent in sentences:
            for pattern in goal_patterns:
                matches = re.findall(pattern, sent, re.IGNORECASE)
                for match in matches:
                    goal = match.strip()
                    if 3 < len(goal) < 80:
                        goals.append(goal)

        return list(dict.fromkeys(goals))[:5]

    @staticmethod
    def _select_important_sentences(
        sentences: list[str], max_sentences: int = 3
    ) -> list[str]:
        """Select the most informative sentences."""
        scored = []
        importance_words = {
            "love", "enjoy", "passionate", "looking", "want", "dream",
            "career", "work", "family", "travel", "adventure", "value",
            "important", "believe", "goal", "partner", "relationship",
        }

        for sent in sentences:
            words = set(sent.lower().split())
            score = len(words & importance_words)
            # Bonus for longer (more informative) sentences
            if len(sent) > 30:
                score += 1
            scored.append((score, sent))

        scored.sort(reverse=True)
        return [s for _, s in scored[:max_sentences] if _]  # Only non-zero scores

    # ── Fallback ────────────────────────────────────────────────────────

    async def _fallback(self, input_data: dict[str, Any], error: Exception) -> dict[str, Any]:
        text = input_data.get("normalized_text", "") or input_data.get("text", "")
        # Ultra simple: first 200 chars as summary
        summary = text[:200].strip()
        return {
            "summary": summary,
            "key_topics": [],
            "stated_preferences": [],
            "relationship_goals": [],
            "token_count": len(summary) // 4,
            "fallback_reason": str(error),
        }
