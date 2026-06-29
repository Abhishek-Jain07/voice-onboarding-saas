"""
Conversation Segmentation Service
──────────────────────────────────
Takes the full transcript + word-level timestamps from Faster-Whisper
and segments it into individual question-answer pairs.

Pipeline:
  1. Receive transcript text + word timestamps
  2. Detect each of the 35 known questions using fuzzy matching
  3. Map each detected question to the subsequent user answer
  4. Output an array of segments with question_id, question, answer, timestamps
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from app.questions import QUESTIONS, OnboardingQuestion
from app.services.base import BaseService


class ConversationSegmentationService(BaseService):
    name = "conversation_segmentation"

    def __init__(self) -> None:
        super().__init__()
        # Pre-compute normalised question texts for matching
        self._questions = QUESTIONS
        self._normalised_questions = [
            self._normalise(q.text) for q in self._questions
        ]

    # ── Text normalisation ───────────────────────────────────────────

    @staticmethod
    def _normalise(text: str) -> str:
        """Lowercase, strip punctuation, collapse whitespace."""
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text

    # ── Fuzzy matching ───────────────────────────────────────────────

    def _match_question(
        self, text: str, threshold: float = 0.55
    ) -> OnboardingQuestion | None:
        """
        Find the best-matching onboarding question for a given text segment.
        Uses SequenceMatcher ratio — no external dependencies needed.
        Returns the question if similarity >= threshold, else None.
        """
        normalised = self._normalise(text)
        if len(normalised) < 3:
            return None

        best_score = 0.0
        best_question = None

        for q, norm_q in zip(self._questions, self._normalised_questions):
            # Quick check: if the normalised question is a substring, high match
            if norm_q in normalised or normalised in norm_q:
                score = 0.95
            else:
                score = SequenceMatcher(None, normalised, norm_q).ratio()

            if score > best_score:
                best_score = score
                best_question = q

        if best_score >= threshold:
            return best_question
        return None

    # ── Sentence splitting from word timestamps ──────────────────────

    @staticmethod
    def _build_sentences_from_words(words: list[dict]) -> list[dict]:
        """
        Group word-level timestamps into sentence-level segments.
        Uses punctuation boundaries (., ?, !) to split sentences.
        Returns list of { text, start, end, words }.
        """
        if not words:
            return []

        sentences = []
        current_words = []
        current_text_parts = []

        for w in words:
            word_text = w.get("word", "").strip()
            if not word_text:
                continue

            current_words.append(w)
            current_text_parts.append(word_text)

            # Check if this word ends a sentence
            if word_text[-1] in ".?!":
                sentences.append({
                    "text": " ".join(current_text_parts),
                    "start": current_words[0].get("start", 0.0),
                    "end": current_words[-1].get("end", 0.0),
                    "words": list(current_words),
                })
                current_words = []
                current_text_parts = []

        # Flush any remaining words as a sentence
        if current_words:
            sentences.append({
                "text": " ".join(current_text_parts),
                "start": current_words[0].get("start", 0.0),
                "end": current_words[-1].get("end", 0.0),
                "words": list(current_words),
            })

        return sentences

    # ── Core segmentation ────────────────────────────────────────────

    async def _process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        Segment the transcript into question-answer pairs.

        Input:
            text: str — full transcript
            words: list[dict] — word-level timestamps from Faster-Whisper
                  each word: { word, start, end, probability }

        Output:
            segments: list[dict] — each segment:
                { question_id, question_text, question_category, question_weight,
                  answer_text, start_time, end_time, word_count }
            unmatched_text: str — any text that couldn't be assigned to a question
            questions_detected: int — how many of the 35 questions were found
        """
        words = input_data.get("words", [])
        full_text = input_data.get("text", "")

        # If no word-level timestamps, fall back to text-only segmentation
        if not words:
            return await self._segment_text_only(full_text)

        # Build sentence-level segments from word timestamps
        sentences = self._build_sentences_from_words(words)

        if not sentences:
            return await self._segment_text_only(full_text)

        # ── Pass 1: Identify which sentences are questions ───────────
        question_positions = []  # (sentence_index, OnboardingQuestion)
        matched_question_ids = set()

        for i, sent in enumerate(sentences):
            q = self._match_question(sent["text"])
            if q and q.id not in matched_question_ids:
                question_positions.append((i, q))
                matched_question_ids.add(q.id)

        # If no questions were detected at all, treat entire transcript as one answer
        if not question_positions:
            self.logger.warning(
                "No questions detected in transcript, treating as single answer"
            )
            return {
                "segments": [{
                    "question_id": 0,
                    "question_text": "Full Conversation",
                    "question_category": "unknown",
                    "question_weight": 1.0,
                    "answer_text": full_text,
                    "start_time": sentences[0]["start"] if sentences else 0.0,
                    "end_time": sentences[-1]["end"] if sentences else 0.0,
                    "word_count": len(full_text.split()),
                }],
                "unmatched_text": "",
                "questions_detected": 0,
            }

        # ── Pass 2: Extract answers between questions ────────────────
        segments = []
        unmatched_parts = []

        # Any text before the first question is unmatched
        first_q_idx = question_positions[0][0]
        if first_q_idx > 0:
            pre_text = " ".join(s["text"] for s in sentences[:first_q_idx])
            if pre_text.strip():
                unmatched_parts.append(pre_text.strip())

        for pos_idx, (sent_idx, question) in enumerate(question_positions):
            # Determine the range of sentences that form the answer
            # Answer starts at the sentence AFTER the question
            answer_start_idx = sent_idx + 1

            # Answer ends at the sentence BEFORE the next question (or end of transcript)
            if pos_idx + 1 < len(question_positions):
                next_q_idx = question_positions[pos_idx + 1][0]
                answer_end_idx = next_q_idx
            else:
                answer_end_idx = len(sentences)

            # Extract answer sentences
            answer_sentences = sentences[answer_start_idx:answer_end_idx]
            answer_text = " ".join(s["text"] for s in answer_sentences).strip()

            # Calculate timestamps
            if answer_sentences:
                start_time = answer_sentences[0]["start"]
                end_time = answer_sentences[-1]["end"]
            else:
                # No answer sentences — use the question's end time
                start_time = sentences[sent_idx]["end"]
                end_time = start_time

            segments.append({
                "question_id": question.id,
                "question_text": question.text,
                "question_category": question.category,
                "question_weight": question.weight,
                "answer_text": answer_text,
                "start_time": round(start_time, 3),
                "end_time": round(end_time, 3),
                "word_count": len(answer_text.split()) if answer_text else 0,
            })

        return {
            "segments": segments,
            "unmatched_text": " ".join(unmatched_parts),
            "questions_detected": len(question_positions),
        }

    # ── Text-only fallback segmentation ──────────────────────────────

    async def _segment_text_only(self, full_text: str) -> dict[str, Any]:
        """
        When word timestamps are not available, split the text using
        question detection on sentence boundaries.
        """
        if not full_text.strip():
            return {
                "segments": [],
                "unmatched_text": "",
                "questions_detected": 0,
            }

        # Split text into rough sentences
        raw_sentences = re.split(r"(?<=[.?!])\s+", full_text)
        sentences = [{"text": s.strip(), "start": 0.0, "end": 0.0}
                     for s in raw_sentences if s.strip()]

        question_positions = []
        matched_ids = set()

        for i, sent in enumerate(sentences):
            q = self._match_question(sent["text"])
            if q and q.id not in matched_ids:
                question_positions.append((i, q))
                matched_ids.add(q.id)

        if not question_positions:
            return {
                "segments": [{
                    "question_id": 0,
                    "question_text": "Full Conversation",
                    "question_category": "unknown",
                    "question_weight": 1.0,
                    "answer_text": full_text,
                    "start_time": 0.0,
                    "end_time": 0.0,
                    "word_count": len(full_text.split()),
                }],
                "unmatched_text": "",
                "questions_detected": 0,
            }

        segments = []
        for pos_idx, (sent_idx, question) in enumerate(question_positions):
            answer_start = sent_idx + 1
            if pos_idx + 1 < len(question_positions):
                answer_end = question_positions[pos_idx + 1][0]
            else:
                answer_end = len(sentences)

            answer_text = " ".join(
                s["text"] for s in sentences[answer_start:answer_end]
            ).strip()

            segments.append({
                "question_id": question.id,
                "question_text": question.text,
                "question_category": question.category,
                "question_weight": question.weight,
                "answer_text": answer_text,
                "start_time": 0.0,
                "end_time": 0.0,
                "word_count": len(answer_text.split()) if answer_text else 0,
            })

        return {
            "segments": segments,
            "unmatched_text": "",
            "questions_detected": len(question_positions),
        }

    # ── Fallback ─────────────────────────────────────────────────────

    async def _fallback(
        self, input_data: dict[str, Any], error: Exception
    ) -> dict[str, Any]:
        """Return the entire transcript as a single segment."""
        full_text = input_data.get("text", "")
        return {
            "segments": [{
                "question_id": 0,
                "question_text": "Full Conversation",
                "question_category": "unknown",
                "question_weight": 1.0,
                "answer_text": full_text,
                "start_time": 0.0,
                "end_time": 0.0,
                "word_count": len(full_text.split()) if full_text else 0,
            }],
            "unmatched_text": "",
            "questions_detected": 0,
            "fallback_reason": str(error),
        }
