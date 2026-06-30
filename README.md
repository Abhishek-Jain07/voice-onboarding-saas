# 🎙️ AI Voice Onboarding System – Backend

A FastAPI-powered backend that analyses voice recordings to generate AI dating profiles. It extracts audio features, emotions, sentiment, interests, personality traits, and more from voice data, then uses an LLM to create a comprehensive dating profile.

## 🏗️ Architecture

```
FastAPI Gateway
  └─ Pipeline Orchestrator
       ├─ Sequential: Audio Preprocessing → STT → Transcript Normalization
       ├─ Parallel:   Audio Features | Emotion | Sentiment | Interests |
       │              Personality | Keywords & Entities | Conversation Summary
       └─ Sequential: Profile Aggregation → Memory → Prompt Builder →
                      LLM Provider → Dating Profile → Response Formatter
```

## 📂 Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                       # FastAPI app, CORS, routes
│   ├── config.py                     # Pydantic settings, env vars
│   ├── logging_config.py             # Structured JSON logging
│   ├── pipeline/
│   │   └── orchestrator.py           # 3-phase pipeline orchestrator
│   ├── services/
│   │   ├── base.py                   # Abstract BaseService
│   │   ├── audio_preprocessing.py    # VAD, noise reduction, normalization
│   │   ├── speech_to_text.py         # Faster-Whisper STT
│   │   ├── transcript_normalization.py  # Filler removal, punctuation
│   │   ├── audio_features.py         # Pitch, energy, speaking rate
│   │   ├── emotion.py                # Rule-based emotion classification
│   │   ├── sentiment.py              # DistilBERT sentiment analysis
│   │   ├── interest_extraction.py    # Zero-shot interest extraction
│   │   ├── personality.py            # Big Five personality traits
│   │   ├── keyword_entity.py         # spaCy NER, keywords, topics
│   │   ├── conversation_summary.py   # Structured summary generator
│   │   ├── profile_aggregation.py    # Merge all service outputs
│   │   ├── memory.py                 # Versioned profile storage
│   │   ├── embedding.py              # Optional sentence embeddings
│   │   ├── prompt_builder.py         # Compact LLM prompt (<500 tokens)
│   │   ├── llm_provider.py           # Multi-provider LLM adapter
│   │   ├── dating_profile.py         # Parse LLM → structured profile
│   │   └── response_formatter.py     # Final response assembly
│   └── models/
│       └── schemas.py                # All Pydantic models
├── requirements.txt
├── Dockerfile
├── railway.json
├── .env.example
└── README.md
```

## 🚀 Quick Start

### 1. Clone & Setup

```bash
cd backend
cp .env.example .env
# Edit .env with your API keys
```

### 2. Install Dependencies

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Run the Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open the Docs

Visit [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive Swagger UI.

## 📡 API Endpoints

### `POST /api/process`

Process an audio file through the full AI pipeline.

**Form Data:**
| Field      | Type   | Required | Description                        |
|------------|--------|----------|------------------------------------|
| `audio`    | File   | ✅       | Audio file (wav, mp3, webm, etc.)  |
| `api_key`  | string | ❌       | LLM provider API key               |
| `provider` | string | ❌       | Provider: openai, gemini, anthropic, ollama, openrouter |
| `model`    | string | ❌       | Model name override                |
| `session_id` | string | ❌     | Session ID for profile versioning  |

**Example:**
```bash
curl -X POST http://localhost:8000/api/process \
  -F "audio=@recording.wav" \
  -F "provider=openai" \
  -F "api_key=sk-..."
```

### `GET /api/health`

Health check for all services.

### `GET /api/providers`

List available LLM providers with their default models.

## 🔧 Supported LLM Providers

| Provider    | API Key Required | Default Model            |
|-------------|-----------------|--------------------------|
| OpenAI      | ✅              | gpt-4o-mini              |
| Gemini      | ✅              | gemini-2.0-flash         |
| Anthropic   | ✅              | claude-sonnet-4-20250514        |
| Ollama      | ❌              | llama3.2                 |
| OpenRouter  | ✅              | meta-llama/llama-3-8b    |

## 🎯 Fallback System

Every service has a fallback mode that activates when ML models aren't available:

| Service                | Primary           | Fallback                    |
|------------------------|-------------------|-----------------------------|
| Audio Preprocessing    | librosa + pydub   | Raw audio passthrough       |
| Speech-to-Text         | faster-whisper    | Mock transcript             |
| Sentiment Analysis     | DistilBERT        | Keyword-based scoring       |
| Interest Extraction    | BART zero-shot    | Keyword matching            |
| Personality Analysis   | BART zero-shot    | Linguistic pattern rules    |
| Keyword & Entity       | spaCy NER         | Regex NER                   |
| Emotion Classification | –                 | Rule-based (always active)  |
| LLM Provider           | HTTP API call     | Mock dating profile         |

## 🐳 Docker

```bash
docker build -t voice-onboarding-backend .
docker run -p 8000:8000 --env-file .env voice-onboarding-backend
```

## 🚂 Railway Deployment

1. Connect your GitHub repo to Railway
2. Set environment variables in the Railway dashboard
3. Railway will auto-detect the `railway.json` and `Dockerfile`

## 📊 Pipeline Output

The response includes:
- **Transcript** – Raw and normalized text with word timestamps
- **Audio Features** – Pitch, energy, speaking rate, voice stability
- **Emotion** – Primary/secondary emotion with confidence scores
- **Sentiment** – Overall and per-sentence sentiment analysis
- **Interests** – Categorized interests (hobbies, food, travel, etc.)
- **Personality** – Big Five traits, communication style, attachment style
- **Keywords & Entities** – Named entities, topics, intent
- **Conversation Summary** – Key topics, preferences, goals
- **Dating Profile** – Bio, compatibility features, ice breakers, flags
- **Timings** – Per-step execution timing in milliseconds

## 📝 License

MIT
