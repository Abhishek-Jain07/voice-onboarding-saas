"""
Pydantic models / schemas for the AI Voice Onboarding System.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════

class LLMProvider(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"


class EmotionLabel(str, Enum):
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    NEUTRAL = "neutral"
    SURPRISED = "surprised"
    EXCITED = "excited"
    CALM = "calm"


class SentimentLabel(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"


# ═══════════════════════════════════════════════════════════════════════
# Request / Response
# ═══════════════════════════════════════════════════════════════════════

class ProcessRequest(BaseModel):
    """Metadata sent alongside the audio file."""
    api_key: Optional[str] = None
    provider: LLMProvider = LLMProvider.OPENAI
    model: Optional[str] = None
    session_id: Optional[str] = None


class ServiceMeta(BaseModel):
    service: str
    duration_ms: float
    fallback: bool = False
    error: Optional[str] = None


class HealthStatus(BaseModel):
    service: str
    status: str = "healthy"
    fallback_mode: bool = False


class HealthResponse(BaseModel):
    status: str
    services: list[HealthStatus]


class ProviderInfo(BaseModel):
    name: str
    requires_api_key: bool
    default_model: str


class ProvidersResponse(BaseModel):
    providers: list[ProviderInfo]


# ═══════════════════════════════════════════════════════════════════════
# Audio Preprocessing
# ═══════════════════════════════════════════════════════════════════════

class AudioInfo(BaseModel):
    sample_rate: int
    duration_seconds: float
    channels: int
    format: str
    rms_db: Optional[float] = None


# ═══════════════════════════════════════════════════════════════════════
# Speech-to-Text
# ═══════════════════════════════════════════════════════════════════════

class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float
    probability: float


class TranscriptResult(BaseModel):
    text: str
    language: str = "en"
    language_probability: float = 0.0
    confidence: float = 0.0
    words: list[WordTimestamp] = []
    duration_seconds: float = 0.0


# ═══════════════════════════════════════════════════════════════════════
# Transcript Normalization
# ═══════════════════════════════════════════════════════════════════════

class NormalizedTranscript(BaseModel):
    original_text: str
    normalized_text: str
    sentences: list[str] = []
    filler_words_removed: int = 0
    language: str = "en"


# ═══════════════════════════════════════════════════════════════════════
# Audio Features
# ═══════════════════════════════════════════════════════════════════════

class AudioFeatures(BaseModel):
    pitch_mean: float = 0.0
    pitch_std: float = 0.0
    pitch_min: float = 0.0
    pitch_max: float = 0.0
    energy_mean: float = 0.0
    energy_std: float = 0.0
    speaking_rate_sps: float = 0.0  # syllables per second
    pause_count: int = 0
    pause_duration_mean: float = 0.0
    pause_duration_total: float = 0.0
    hesitation_count: int = 0
    voice_stability: float = 0.0  # 0–1, higher = more stable
    speaking_style: str = "conversational"


# ═══════════════════════════════════════════════════════════════════════
# Emotion
# ═══════════════════════════════════════════════════════════════════════

class EmotionResult(BaseModel):
    primary_emotion: str = "neutral"
    secondary_emotion: Optional[str] = None
    confidence: float = 0.0
    emotion_scores: dict[str, float] = {}


# ═══════════════════════════════════════════════════════════════════════
# Sentiment
# ═══════════════════════════════════════════════════════════════════════

class SentenceSentiment(BaseModel):
    text: str
    label: str
    score: float


class SentimentResult(BaseModel):
    overall_label: str = "NEUTRAL"
    overall_score: float = 0.0
    sentences: list[SentenceSentiment] = []


# ═══════════════════════════════════════════════════════════════════════
# Interest Extraction
# ═══════════════════════════════════════════════════════════════════════

class InterestCategory(BaseModel):
    category: str
    items: list[str] = []
    confidence: float = 0.0


class InterestResult(BaseModel):
    interests: list[InterestCategory] = []
    raw_labels: dict[str, float] = {}


# ═══════════════════════════════════════════════════════════════════════
# Personality
# ═══════════════════════════════════════════════════════════════════════

class PersonalityTraits(BaseModel):
    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5


class PersonalityResult(BaseModel):
    big_five: PersonalityTraits = PersonalityTraits()
    communication_style: str = "balanced"
    attachment_style: str = "secure"
    confidence: float = 0.0


# ═══════════════════════════════════════════════════════════════════════
# Keyword & Entity
# ═══════════════════════════════════════════════════════════════════════

class NamedEntity(BaseModel):
    text: str
    label: str  # PERSON, ORG, GPE, …
    start: int = 0
    end: int = 0


class KeywordEntityResult(BaseModel):
    entities: list[NamedEntity] = []
    keywords: list[str] = []
    topics: list[str] = []
    intent: str = "general"


# ═══════════════════════════════════════════════════════════════════════
# Conversation Summary
# ═══════════════════════════════════════════════════════════════════════

class ConversationSummary(BaseModel):
    summary: str = ""
    key_topics: list[str] = []
    stated_preferences: list[str] = []
    relationship_goals: list[str] = []
    token_count: int = 0


# ═══════════════════════════════════════════════════════════════════════
# Profile Aggregation
# ═══════════════════════════════════════════════════════════════════════

class AggregatedProfile(BaseModel):
    audio_features: Optional[AudioFeatures] = None
    emotion: Optional[EmotionResult] = None
    sentiment: Optional[SentimentResult] = None
    interests: Optional[InterestResult] = None
    personality: Optional[PersonalityResult] = None
    keywords_entities: Optional[KeywordEntityResult] = None
    conversation_summary: Optional[ConversationSummary] = None
    transcript: Optional[NormalizedTranscript] = None
    cross_confidence: float = 0.0


# ═══════════════════════════════════════════════════════════════════════
# Memory / Profile Versioning
# ═══════════════════════════════════════════════════════════════════════

class ProfileVersion(BaseModel):
    version: int = 1
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    profile: AggregatedProfile = AggregatedProfile()
    confidence: float = 0.0


class ProfileMemory(BaseModel):
    session_id: str
    versions: list[ProfileVersion] = []
    current_version: int = 0


# ═══════════════════════════════════════════════════════════════════════
# Dating Profile
# ═══════════════════════════════════════════════════════════════════════

class DatingProfile(BaseModel):
    personality_summary: str = ""
    dating_bio: str = ""
    compatibility_features: list[str] = []
    ice_breakers: list[str] = []
    green_flags: list[str] = []
    red_flags: list[str] = []
    conversation_style: str = ""
    match_recommendations: list[str] = []


# ═══════════════════════════════════════════════════════════════════════
# Final Response
# ═══════════════════════════════════════════════════════════════════════

class PipelineTimings(BaseModel):
    total_ms: float = 0.0
    steps: dict[str, float] = {}


class ProcessResponse(BaseModel):
    success: bool = True
    session_id: str = ""
    transcript: Optional[TranscriptResult] = None
    normalized_transcript: Optional[NormalizedTranscript] = None
    audio_features: Optional[AudioFeatures] = None
    emotion: Optional[EmotionResult] = None
    sentiment: Optional[SentimentResult] = None
    interests: Optional[InterestResult] = None
    personality: Optional[PersonalityResult] = None
    keywords_entities: Optional[KeywordEntityResult] = None
    conversation_summary: Optional[ConversationSummary] = None
    aggregated_profile: Optional[AggregatedProfile] = None
    dating_profile: Optional[DatingProfile] = None
    llm_prompt: Optional[str] = None
    llm_raw_response: Optional[str] = None
    timings: PipelineTimings = PipelineTimings()
    errors: list[str] = []
    service_meta: list[ServiceMeta] = []
