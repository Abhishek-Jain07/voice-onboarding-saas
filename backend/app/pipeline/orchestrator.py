"""
Pipeline Orchestrator
- Sequential: Audio Preprocessing → STT → Transcript Normalization
- Parallel: 7 AI analysis services concurrently via asyncio.gather
- Sequential: Aggregation → Memory → Prompt → LLM → Profile → Format
- Per-step timing
- Graceful error handling (parallel failures don't kill pipeline)
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
        # PHASE 1 – Sequential: Audio → STT → Normalization
        # ════════════════════════════════════════════════════════════════

        logger.info("Phase 1: Sequential audio processing", extra={"step": "phase1"})

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

        # ════════════════════════════════════════════════════════════════
        # PHASE 2 – Parallel: 7 AI analysis services
        # ════════════════════════════════════════════════════════════════

        logger.info("Phase 2: Parallel AI analysis", extra={"step": "phase2"})
        step_start = time.perf_counter()

        # Build shared input for parallel services
        parallel_input_audio = {
            "processed_audio": preprocess_result.get("processed_audio", audio_bytes),
            "sample_rate": preprocess_result.get("sample_rate", 16000),
        }
        parallel_input_text = {
            "text": stt_result.get("text", ""),
            "normalized_text": norm_result.get("normalized_text", ""),
            "sentences": norm_result.get("sentences", []),
            "language": norm_result.get("language", "en"),
        }

        # Launch all 7 services concurrently
        parallel_tasks = [
            self._safe_service_call("audio_features", self.audio_features, parallel_input_audio),
            self._safe_service_call("emotion", self.emotion, parallel_input_audio),
            self._safe_service_call("sentiment", self.sentiment, parallel_input_text),
            self._safe_service_call("interest_extraction", self.interest_extraction, parallel_input_text),
            self._safe_service_call("personality", self.personality, parallel_input_text),
            self._safe_service_call("keyword_entity", self.keyword_entity, parallel_input_text),
            self._safe_service_call("conversation_summary", self.conversation_summary, parallel_input_text),
        ]

        parallel_results = await asyncio.gather(*parallel_tasks)
        timings["parallel_analysis"] = round((time.perf_counter() - step_start) * 1000, 2)

        # Unpack results
        (
            audio_feat_result,
            emotion_result,
            sentiment_result,
            interest_result,
            personality_result,
            keyword_entity_result,
            summary_result,
        ) = parallel_results

        # Collect meta and errors
        for name, result in [
            ("audio_features", audio_feat_result),
            ("emotion", emotion_result),
            ("sentiment", sentiment_result),
            ("interest_extraction", interest_result),
            ("personality", personality_result),
            ("keyword_entity", keyword_entity_result),
            ("conversation_summary", summary_result),
        ]:
            meta = result.get("_meta", {})
            service_meta.append(meta)
            if meta.get("error"):
                errors.append(f"{name}: {meta['error']}")

        # Emotion service may need audio features – run again if needed
        if audio_feat_result and not emotion_result.get("_meta", {}).get("fallback"):
            pass  # Already ran fine
        elif audio_feat_result:
            # Re-run emotion with actual audio features
            emotion_input = {**audio_feat_result}
            emotion_input.pop("_meta", None)
            emotion_result = await self.emotion.process(emotion_input)

        state.update({
            "audio_features": audio_feat_result,
            "emotion": emotion_result,
            "sentiment": sentiment_result,
            "interests": interest_result,
            "personality": personality_result,
            "keywords_entities": keyword_entity_result,
            "conversation_summary": summary_result,
        })

        # ════════════════════════════════════════════════════════════════
        # PHASE 3 – Sequential: Aggregate → Memory → Prompt → LLM → Profile
        # ════════════════════════════════════════════════════════════════

        logger.info("Phase 3: Aggregation and profile generation", extra={"step": "phase3"})

        # Step: Profile Aggregation
        step_start = time.perf_counter()
        agg_result = await self.profile_aggregation.process({
            "audio_features": audio_feat_result,
            "emotion": emotion_result,
            "sentiment": sentiment_result,
            "interests": interest_result,
            "personality": personality_result,
            "keywords_entities": keyword_entity_result,
            "conversation_summary": summary_result,
            "normalized_transcript": norm_result,
        })
        timings["profile_aggregation"] = round((time.perf_counter() - step_start) * 1000, 2)
        service_meta.append(agg_result.get("_meta", {}))
        state["aggregated_profile"] = agg_result

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
            f"Pipeline completed in {total_ms:.0f}ms",
            extra={"step": "complete", "duration_ms": total_ms},
        )

        return final_result

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
