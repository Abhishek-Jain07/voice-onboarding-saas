"""
Onboarding Question Bank
─────────────────────────
All 35 predefined interview questions with:
- Unique ID
- Question text
- Category (for grouping)
- Weight (for weighted aggregation — higher = more important for profile)

Weight guide:
  0.5 – Simple factual questions (age, name, occupation)
  1.0 – Lifestyle questions
  1.2 – Hobbies & interests
  1.5 – Self-introduction / open-ended
  2.0 – Relationship goals, values & beliefs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class OnboardingQuestion:
    id: int
    text: str
    category: str
    weight: float


# ── The 35 Onboarding Questions ─────────────────────────────────────────

QUESTIONS: List[OnboardingQuestion] = [
    # ── Self Introduction ────────────────────────────────────────────
    OnboardingQuestion(1,  "Tell me about yourself", "self_introduction", 1.5),
    OnboardingQuestion(2,  "What is your name", "factual", 0.5),
    OnboardingQuestion(3,  "How old are you", "factual", 0.5),
    OnboardingQuestion(4,  "Where are you from", "factual", 0.5),
    OnboardingQuestion(5,  "What do you do for a living", "factual", 0.5),

    # ── Hobbies & Interests ──────────────────────────────────────────
    OnboardingQuestion(6,  "What do you enjoy doing on weekends", "hobbies", 1.2),
    OnboardingQuestion(7,  "What are your hobbies", "hobbies", 1.2),
    OnboardingQuestion(8,  "What kind of music do you like", "hobbies", 1.0),
    OnboardingQuestion(9,  "What kind of movies or shows do you watch", "hobbies", 1.0),
    OnboardingQuestion(10, "Do you enjoy traveling", "hobbies", 1.2),
    OnboardingQuestion(11, "What is your favorite travel destination", "hobbies", 1.0),
    OnboardingQuestion(12, "Do you enjoy cooking", "hobbies", 1.0),
    OnboardingQuestion(13, "Do you play any sports", "hobbies", 1.0),
    OnboardingQuestion(14, "What kind of books do you read", "hobbies", 1.0),

    # ── Lifestyle ────────────────────────────────────────────────────
    OnboardingQuestion(15, "Describe your typical day", "lifestyle", 1.0),
    OnboardingQuestion(16, "Are you a morning person or night owl", "lifestyle", 1.0),
    OnboardingQuestion(17, "How do you spend your free time", "lifestyle", 1.0),
    OnboardingQuestion(18, "Do you prefer going out or staying in", "lifestyle", 1.0),
    OnboardingQuestion(19, "Do you have any pets", "lifestyle", 0.5),
    OnboardingQuestion(20, "How important is fitness to you", "lifestyle", 1.0),

    # ── Relationship Goals ───────────────────────────────────────────
    OnboardingQuestion(21, "What are you looking for in a partner", "relationship", 2.0),
    OnboardingQuestion(22, "What does your ideal relationship look like", "relationship", 2.0),
    OnboardingQuestion(23, "What is your relationship deal breaker", "relationship", 2.0),
    OnboardingQuestion(24, "How important is communication in a relationship", "relationship", 2.0),
    OnboardingQuestion(25, "What is your love language", "relationship", 2.0),

    # ── Values & Beliefs ─────────────────────────────────────────────
    OnboardingQuestion(26, "What values are most important to you", "values", 2.0),
    OnboardingQuestion(27, "How important is family to you", "values", 2.0),
    OnboardingQuestion(28, "What does success mean to you", "values", 1.5),
    OnboardingQuestion(29, "How do you handle conflict", "values", 2.0),
    OnboardingQuestion(30, "What role does faith or spirituality play in your life", "values", 1.5),

    # ── Personality & Self-Reflection ────────────────────────────────
    OnboardingQuestion(31, "How would your friends describe you", "self_introduction", 1.5),
    OnboardingQuestion(32, "What is your biggest strength", "self_introduction", 1.5),
    OnboardingQuestion(33, "What is something you are passionate about", "self_introduction", 1.5),
    OnboardingQuestion(34, "What makes you happy", "self_introduction", 1.5),
    OnboardingQuestion(35, "Is there anything else you want a potential match to know", "relationship", 2.0),
]


# ── Lookup helpers ───────────────────────────────────────────────────────

QUESTION_BY_ID: dict[int, OnboardingQuestion] = {q.id: q for q in QUESTIONS}
QUESTION_TEXTS: list[str] = [q.text.lower() for q in QUESTIONS]


def get_question(question_id: int) -> OnboardingQuestion | None:
    """Return the question for a given ID, or None."""
    return QUESTION_BY_ID.get(question_id)


def get_weight(question_id: int) -> float:
    """Return the weight for a given question ID, defaulting to 1.0."""
    q = QUESTION_BY_ID.get(question_id)
    return q.weight if q else 1.0


def get_category(question_id: int) -> str:
    """Return the category for a given question ID."""
    q = QUESTION_BY_ID.get(question_id)
    return q.category if q else "unknown"
