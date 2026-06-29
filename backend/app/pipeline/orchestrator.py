"""
Pipeline Orchestrator (v2 – Per-Question Analysis)
───────────────────────────────────────────────────
Phase 1: Audio Preprocessing → STT → Normalization → **Segmentation**
Phase 2: For each user answer → 7 AI services in parallel
Phase 3: Score Aggregation → Memory → Prompt → LLM → Profile → Format
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Optional

from app.logging_config import get_logger
from app.services.audio_preprocessing import AudioPreprocessingService
from app.services.speech_to_text import SpeechToTextService
from app.services.transcript_normalization import TranscriptNormalizationService
from app.services.segmentation import ConversationSegmentationService
from app.services.audio_features import AudioFeaturesService
from app.services.emotion import EmotionService
from app.services.sentiment import SentimentService
from app.services.interest_extraction import InterestExtractionService
from app.services.personality import PersonalityService
from app.services.keyword_entity import KeywordEntityService
from app.services.conversation_summary import ConversationSummaryService
from app.services.profile_aggregation import ProfileAggregationService
from app.services.memory import MemoryService
from app.services.prompt_builder import PromptBuilderService
from app.services.llm_provider import LLMProviderService
from app.services.dating_profile import DatingProfileService
from app.services.response_formatter import ResponseFormatterService

logger = get_logger("pipeline.orchestrator")


class PipelineOrchestrator:
    """Manages the full voice-analysis → dating-profile pipeline."""

    def __init__(self) -> None:
        # Instantiate all services
        self.audio_preprocess = AudioPreprocessingService()
        self.stt = SpeechToTextService()
        self.transcript_norm = TranscriptNormalizationService()
        self.segmentation = ConversationSegmentationService()
        self.audio_features = AudioFeaturesService()
        self.emotion = EmotionService()
        self.sentiment = SentimentService()
        self.interest_extraction = InterestExtractionService()
        self.personality = PersonalityService()
        self.keyword_entity = KeywordEntityService()
        self.conversation_summary = ConversationSummaryService()
        self.profile_aggregation = ProfileAggregationService()
        self.memory = MemoryService()
        self.prompt_builder = PromptBuilderService()
        self.llm_provider = LLMProviderService()
        self.dating_profile = DatingProfileService()
        self.response_formatter = ResponseFormatterService()

        # All services for health checks
        self._all_services = [
            self.audio_preprocess,
            self.stt,
            self.transcript_norm,
            self.segmentation,
            self.audio_features,
            self.emotion,
            self.sentiment,
            self.interest_extraction,
            self.personality,
            self.keyword_entity,
            self.conversation_summary,
            self.profile_aggregation,
            self.memory,
            self.prompt_builder,
            self.llm_provider,
            self.dating_profile,
            self.response_formatter,
        ]

    # ── Main Pipeline ───────────────────────────────────────────────────

    async def run(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        api_key: Optional[str] = None,
        provider: str = "openai",
        model: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Execute the full pipeline and return formatted results."""

        pipeline_start = time.perf_counter()
        session_id = session_id or str(uuid.uuid4())
        timings: dict[str, float] = {}
        service_meta: list[dict] = []
        errors: list[str] = []

        # Container for pipeline state
        state: dict[str, Any] = {
            "session_id": session_id,
        }

        # ════════════════════════════════════════════════════════════════
        # PHASE 1 – Sequential: Audio → STT → Normalization → Segmentation
        # ════════════════════════════════════════════════════════════════

        logger.info("Phase 1: Sequential audio processing + segmentation", extra={"step": "phase1"})

        # Step 1: Audio Preprocessing
        step_start = time.perf_counter()
        preprocess_result = await self.audio_preprocess.process({
            "audio_bytes": audio_bytes,
            "filename": filename,
        })
        timings["audio_preprocessing"] = round((time.perf_counter() - step_start) * 1000, 2)
        service_meta.append(preprocess_result.get("_meta", {}))

        if "error" in preprocess_result and "processed_audio" not in preprocess_result:
            errors.append(f"Audio preprocessing: {preprocess_result['error']}")

        state["preprocess"] = preprocess_result

        # Step 2: Speech-to-Text
        step_start = time.perf_counter()
        stt_result = await self.stt.process({
            "processed_audio": preprocess_result.get("processed_audio", audio_bytes),
            "sample_rate": preprocess_result.get("sample_rate", 16000),
            "duration_seconds": preprocess_result.get("duration_seconds", 0),
        })
        timings["speech_to_text"] = round((time.perf_counter() - step_start) * 1000, 2)
        service_meta.append(stt_result.get("_meta", {}))
        state["transcript"] = stt_result

        # Step 3: Transcript Normalization
        step_start = time.perf_counter()
        norm_result = await self.transcript_norm.process({
            "text": stt_result.get("text", ""),
            "language": stt_result.get("language", "en"),
        })
        timings["transcript_normalization"] = round((time.perf_counter() - step_start) * 1000, 2)
        service_meta.append(norm_result.get("_meta", {}))
        state["normalized_transcript"] = norm_result

        # Step 4: Conversation Segmentation (NEW)
        step_start = time.perf_counter()
        seg_result = await self.segmentation.process({
            "text": stt_result.get("text", ""),
            "words": stt_result.get("words", []),
        })
        timings["segmentation"] = round((time.perf_counter() - step_start) * 1000, 2)
        service_meta.append(seg_result.get("_meta", {}))
        state["segmentation"] = seg_result

        segments = seg_result.get("segments", [])
        questions_detected = seg_result.get("questions_detected", 0)

        logger.info(
            f"Segmentation complete: {questions_detected} questions detected, "
            f"{len(segments)} segments",
            extra={"step": "segmentation", "questions": questions_detected},
        )

        # ════════════════════════════════════════════════════════════════
        # PHASE 2 – Per-Question AI Analysis
        # ════════════════════════════════════════════════════════════════

        logger.info(
            f"Phase 2: Analyzing {len(segments)} answer segments",
            extra={"step": "phase2"},
        )
        step_start = time.perf_counter()

        # Run all segments' analyses concurrently
        segment_tasks = [
            self._analyse_single_answer(segment, preprocess_result)
            for segment in segments
        ]
        question_analyses = await asyncio.gather(*segment_tasks)

        timings["per_question_analysis"] = round(
            (time.perf_counter() - step_start) * 1000, 2
        )

        # Collect per-question errors
        for qa in question_analyses:
            for err in qa.get("_errors", []):
                errors.append(err)

        state["question_analyses"] = question_analyses

        # Also keep backward-compatible top-level results by using the
        # first segment or overall aggregated data
        # (These will be overridden by Phase 3 aggregation)

        # ════════════════════════════════════════════════════════════════
        # PHASE 3 – Aggregation → Memory → Prompt → LLM → Profile
        # ════════════════════════════════════════════════════════════════

        logger.info("Phase 3: Aggregation and profile generation", extra={"step": "phase3"})

        # Step: Profile Aggregation (rewritten for per-question data)
        step_start = time.perf_counter()
        agg_result = await self.profile_aggregation.process({
            "question_analyses": question_analyses,
            "normalized_transcript": norm_result,
            "conversation_text": stt_result.get("text", ""),
        })
        timings["profile_aggregation"] = round((time.perf_counter() - step_start) * 1000, 2)
        service_meta.append(agg_result.get("_meta", {}))
        state["aggregated_profile"] = agg_result

        # Extract backward-compatible top-level fields from aggregation
        state["audio_features"] = agg_result.get("audio_features", {})
        state["emotion"] = agg_result.get("emotion", {})
        state["sentiment"] = agg_result.get("sentiment", {})
        state["interests"] = agg_result.get("interests", {})
        state["personality"] = agg_result.get("personality", {})
        state["keywords_entities"] = agg_result.get("keywords_entities", {})

        # Run conversation summary on the full text (not per-question)
        step_start = time.perf_counter()
        summary_result = await self.conversation_summary.process({
            "text": stt_result.get("text", ""),
            "normalized_text": norm_result.get("normalized_text", ""),
            "sentences": norm_result.get("sentences", []),
            "language": norm_result.get("language", "en"),
        })
        timings["conversation_summary"] = round((time.perf_counter() - step_start) * 1000, 2)
        service_meta.append(summary_result.get("_meta", {}))
        state["conversation_summary"] = summary_result

        # Step: Memory
        step_start = time.perf_counter()
        memory_result = await self.memory.process({
            "session_id": session_id,
            "aggregated_profile": agg_result,
        })
        timings["memory"] = round((time.perf_counter() - step_start) * 1000, 2)
        service_meta.append(memory_result.get("_meta", {}))

        # Step: Prompt Builder
        step_start = time.perf_counter()
        prompt_result = await self.prompt_builder.process({
            "profile": memory_result.get("profile", agg_result),
        })
        timings["prompt_builder"] = round((time.perf_counter() - step_start) * 1000, 2)
        service_meta.append(prompt_result.get("_meta", {}))
        state["llm_prompt"] = prompt_result.get("prompt", "")

        # Step: LLM Provider
        step_start = time.perf_counter()
        llm_result = await self.llm_provider.process({
            "prompt": prompt_result.get("prompt", ""),
            "provider": provider,
            "api_key": api_key,
            "model": model,
        })
        timings["llm_provider"] = round((time.perf_counter() - step_start) * 1000, 2)
        service_meta.append(llm_result.get("_meta", {}))
        state["llm_raw_response"] = llm_result.get("response", "")

        # Step: Dating Profile Generator
        step_start = time.perf_counter()
        dating_result = await self.dating_profile.process({
            "response": llm_result.get("response", ""),
            "profile": memory_result.get("profile", agg_result),
        })
        timings["dating_profile"] = round((time.perf_counter() - step_start) * 1000, 2)
        service_meta.append(dating_result.get("_meta", {}))
        state["dating_profile"] = dating_result

        # Step: Response Formatter
        total_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)
        timings["total"] = total_ms

        step_start = time.perf_counter()
        final_result = await self.response_formatter.process({
            **state,
            "timings": {"total_ms": total_ms, "steps": timings},
            "service_meta": [m for m in service_meta if m],
            "errors": errors,
        })
        timings["response_formatter"] = round((time.perf_counter() - step_start) * 1000, 2)

        logger.info(
            f"Pipeline completed in {total_ms:.0f}ms "
            f"({questions_detected} questions, {len(segments)} segments)",
            extra={"step": "complete", "duration_ms": total_ms},
        )

        return final_result

    # ── Per-Answer Analysis ─────────────────────────────────────────────

    async def _analyse_single_answer(
        self,
        segment: dict[str, Any],
        preprocess_result: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Run all 6 AI analysis services on a single answer segment.
        Returns an enriched segment dict with analysis results.
        """
        answer_text = segment.get("answer_text", "")
        question_id = segment.get("question_id", 0)
        errors = []

        if not answer_text.strip():
            return {
                **segment,
                "emotion": {},
                "sentiment": {},
                "audio_features": {},
                "interests": {},
                "personality": {},
                "keywords_entities": {},
                "_errors": [],
            }

        # Prepare inputs
        text_input = {
            "text": answer_text,
            "normalized_text": answer_text,
            "sentences": [answer_text],
            "language": "en",
        }

        audio_input = {
            "processed_audio": preprocess_result.get("processed_audio", b""),
            "sample_rate": preprocess_result.get("sample_rate", 16000),
        }

        # Launch all 6 services concurrently for this answer
        tasks = [
            self._safe_service_call(f"emotion_q{question_id}", self.emotion, audio_input),
            self._safe_service_call(f"sentiment_q{question_id}", self.sentiment, text_input),
            self._safe_service_call(f"audio_features_q{question_id}", self.audio_features, audio_input),
            self._safe_service_call(f"interests_q{question_id}", self.interest_extraction, text_input),
            self._safe_service_call(f"personality_q{question_id}", self.personality, text_input),
            self._safe_service_call(f"keywords_q{question_id}", self.keyword_entity, text_input),
        ]

        results = await asyncio.gather(*tasks)

        emotion_result, sentiment_result, audio_feat_result, \
            interest_result, personality_result, keyword_result = results

        # Collect errors
        for name, result in [
            ("emotion", emotion_result),
            ("sentiment", sentiment_result),
            ("audio_features", audio_feat_result),
            ("interests", interest_result),
            ("personality", personality_result),
            ("keywords", keyword_result),
        ]:
            meta = result.get("_meta", {})
            if meta.get("error"):
                errors.append(f"Q{question_id}_{name}: {meta['error']}")

        # Strip meta from results
        for r in [emotion_result, sentiment_result, audio_feat_result,
                   interest_result, personality_result, keyword_result]:
            r.pop("_meta", None)
            r.pop("fallback_reason", None)

        return {
            **segment,
            "emotion": emotion_result,
            "sentiment": sentiment_result,
            "audio_features": audio_feat_result,
            "interests": interest_result,
            "personality": personality_result,
            "keywords_entities": keyword_result,
            "_errors": errors,
        }

    # ── Helpers ──────────────────────────────────────────────────────────

    async def _safe_service_call(
        self, name: str, service: Any, input_data: dict
    ) -> dict[str, Any]:
        """Wrapper that catches exceptions so parallel services don't kill others."""
        try:
            return await service.process(input_data)
        except Exception as exc:
            logger.error(
                f"Parallel service {name} failed: {exc}",
                extra={"service": name, "error": str(exc)},
            )
            return {
                "_meta": {
                    "service": name,
                    "duration_ms": 0,
                    "fallback": True,
                    "error": str(exc),
                },
                "error": str(exc),
            }

    async def health_check(self) -> list[dict[str, Any]]:
        """Run health checks on all services."""
        results = []
        for service in self._all_services:
            try:
                status = await service.health_check()
                results.append(status)
            except Exception as exc:
                results.append({
                    "service": service.name,
                    "status": "unhealthy",
                    "error": str(exc),
                })
        return results
