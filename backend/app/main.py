"""
FastAPI Application – AI Voice Onboarding System
- POST /api/process    → full pipeline
- GET  /api/health     → service health
- GET  /api/providers  → available LLM providers
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.logging_config import get_logger, setup_logging
from app.pipeline.orchestrator import PipelineOrchestrator
from app.services.llm_provider import LLMProviderService

# ── Logging ─────────────────────────────────────────────────────────────
setup_logging(settings.log_level)
logger = get_logger("app.main")

# ── Pipeline (singleton) ────────────────────────────────────────────────
orchestrator = PipelineOrchestrator()


# ── Lifespan ────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        f"Starting {settings.app_name} v{settings.app_version}",
        extra={"step": "startup"},
    )
    # Ensure upload & profile dirs exist
    settings.upload_path
    settings.profile_path
    yield
    logger.info("Shutting down", extra={"step": "shutdown"})


# ── App ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI Voice Onboarding System – Analyse voice recordings and generate dating profiles.",
    lifespan=lifespan,
)

# ── CORS ────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════════

@app.post("/api/process")
async def process_audio(
    audio: UploadFile = File(..., description="Audio file (wav, mp3, webm, ogg, m4a, flac)"),
    api_key: Optional[str] = Form(None, description="LLM provider API key"),
    provider: Optional[str] = Form("openai", description="LLM provider name"),
    model: Optional[str] = Form(None, description="LLM model name override"),
    session_id: Optional[str] = Form(None, description="Session ID for profile versioning"),
):
    """
    Process an audio file through the full AI pipeline.

    1. Audio preprocessing (VAD, noise reduction, normalization)
    2. Speech-to-text transcription
    3. Transcript normalization
    4. Parallel AI analysis (emotion, sentiment, interests, personality, keywords, summary, audio features)
    5. Profile aggregation & memory
    6. LLM-powered dating profile generation
    """

    # ── Validate file ────────────────────────────────────────────────────
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    content_type = audio.content_type or ""
    filename = audio.filename

    # Read bytes
    audio_bytes = await audio.read()

    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")

    if len(audio_bytes) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {settings.max_upload_size_mb}MB",
        )

    logger.info(
        f"Processing audio: {filename} ({len(audio_bytes)} bytes)",
        extra={"step": "api_process", "details": {"filename": filename, "size": len(audio_bytes)}},
    )

    # ── Run pipeline ─────────────────────────────────────────────────────
    try:
        result = await orchestrator.run(
            audio_bytes=audio_bytes,
            filename=filename,
            api_key=api_key,
            provider=provider or "openai",
            model=model,
            session_id=session_id,
        )
        return JSONResponse(content=result)
    except Exception as exc:
        logger.error(f"Pipeline error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(exc)}")


@app.get("/api/health")
async def health_check():
    """Health check for all services."""
    try:
        statuses = await orchestrator.health_check()
        all_healthy = all(s.get("status") == "healthy" for s in statuses)
        return {
            "status": "healthy" if all_healthy else "degraded",
            "services": statuses,
        }
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(exc)},
        )


@app.get("/api/providers")
async def list_providers():
    """List available LLM providers."""
    providers = LLMProviderService.get_providers_info()
    return {"providers": providers}


# ── Root ─────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/api/health",
    }
