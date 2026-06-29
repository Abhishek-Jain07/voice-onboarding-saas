"""
Response Formatter Service (v2 – Per-Question Data)
- Format the complete response with all intermediate results + final dating profile
- Include per-question analyses and aggregation statistics
- Collect timing and metadata
"""

from __future__ import annotations

from typing import Any

from app.services.base import BaseService


class ResponseFormatterService(BaseService):
    name = "response_formatter"

    # ── Core Processing ─────────────────────────────────────────────────

    async def _process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        Assemble the final API response from all pipeline outputs.

        Expected input_data keys:
            - session_id
            - transcript (STT result)
            - normalized_transcript
            - segmentation
            - question_analyses (list of per-question results)
            - audio_features (aggregated)
            - emotion (aggregated)
            - sentiment (aggregated)
            - interests (aggregated)
            - personality (aggregated)
            - keywords_entities (aggregated)
            - conversation_summary
            - aggregated_profile
            - dating_profile
            - llm_prompt
            - llm_raw_response
            - timings (dict of step_name → ms)
            - service_meta (list of service meta dicts)
            - errors (list of error strings)
        """

        # Clean up internal keys from sub-results
        def clean(d: Any) -> Any:
            if isinstance(d, dict):
                return {
                    k: clean(v)
                    for k, v in d.items()
                    if k not in ("_meta", "processed_audio", "fallback_reason",
                                 "mock", "_errors")
                }
            if isinstance(d, list):
                return [clean(item) for item in d]
            return d

        # Clean per-question analyses
        question_analyses = input_data.get("question_analyses", [])
        clean_qa = []
        for qa in question_analyses:
            cleaned = clean(qa)
            # Remove internal error list from each question analysis
            cleaned.pop("_errors", None)
            clean_qa.append(cleaned)

        # Extract aggregation stats
        aggregated = input_data.get("aggregated_profile", {})
        aggregation_stats = aggregated.get("aggregation_stats", {})

        response = {
            "success": len(input_data.get("errors", [])) == 0,
            "session_id": input_data.get("session_id", ""),
            "transcript": clean(input_data.get("transcript")),
            "normalized_transcript": clean(input_data.get("normalized_transcript")),
            "audio_features": clean(input_data.get("audio_features")),
            "emotion": clean(input_data.get("emotion")),
            "sentiment": clean(input_data.get("sentiment")),
            "interests": clean(input_data.get("interests")),
            "personality": clean(input_data.get("personality")),
            "keywords_entities": clean(input_data.get("keywords_entities")),
            "conversation_summary": clean(input_data.get("conversation_summary")),
            "aggregated_profile": clean(aggregated),
            "dating_profile": clean(input_data.get("dating_profile")),
            # ── New per-question data ────────────────────────────────
            "question_analyses": clean_qa,
            "aggregation_stats": aggregation_stats,
            # ── Meta ─────────────────────────────────────────────────
            "llm_prompt": input_data.get("llm_prompt"),
            "llm_raw_response": input_data.get("llm_raw_response"),
            "timings": input_data.get("timings", {"total_ms": 0, "steps": {}}),
            "errors": input_data.get("errors", []),
            "service_meta": input_data.get("service_meta", []),
        }

        return response

    # ── Fallback ────────────────────────────────────────────────────────

    async def _fallback(self, input_data: dict[str, Any], error: Exception) -> dict[str, Any]:
        return {
            "success": False,
            "session_id": input_data.get("session_id", ""),
            "errors": [str(error)] + input_data.get("errors", []),
            "timings": input_data.get("timings", {}),
            "service_meta": input_data.get("service_meta", []),
            "question_analyses": [],
            "aggregation_stats": {},
            "fallback_reason": str(error),
        }
