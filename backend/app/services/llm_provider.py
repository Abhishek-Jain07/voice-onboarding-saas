"""
LLM Provider Adapter Service
- Multi-provider support: OpenAI, Google Gemini, Anthropic Claude, Ollama, OpenRouter
- Async HTTP calls via httpx
- Per-request API key support
"""

from __future__ import annotations

import json
from typing import Any, Optional

import httpx

from app.config import settings
from app.services.base import BaseService


class LLMProviderService(BaseService):
    name = "llm_provider"

    PROVIDER_CONFIGS = {
        "openai": {
            "base_url": "https://api.openai.com/v1/chat/completions",
            "default_model": "gpt-4o-mini",
            "requires_api_key": True,
        },
        "gemini": {
            "base_url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            "default_model": "gemini-2.0-flash",
            "requires_api_key": True,
        },
        "anthropic": {
            "base_url": "https://api.anthropic.com/v1/messages",
            "default_model": "claude-sonnet-4-20250514",
            "requires_api_key": True,
        },
        "ollama": {
            "base_url": "{base}/api/chat",
            "default_model": "llama3.2",
            "requires_api_key": False,
        },
        "openrouter": {
            "base_url": "https://openrouter.ai/api/v1/chat/completions",
            "default_model": "meta-llama/llama-3-8b-instruct",
            "requires_api_key": True,
        },
        "minimax": {
            "base_url": "https://integrate.api.nvidia.com/v1/chat/completions",
            "default_model": "minimaxai/minimax-m3",
            "requires_api_key": True,
        },
    }

    # ── Core Processing ─────────────────────────────────────────────────

    async def _process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        prompt: str = input_data["prompt"]
        provider: str = input_data.get("provider", settings.default_llm_provider)
        api_key: Optional[str] = input_data.get("api_key")
        model: Optional[str] = input_data.get("model")

        provider = provider.lower()
        if provider not in self.PROVIDER_CONFIGS:
            raise ValueError(f"Unsupported provider: {provider}")

        config = self.PROVIDER_CONFIGS[provider]
        model = model or config["default_model"]

        # Resolve API key
        if not api_key:
            api_key = self._get_default_key(provider)

        if config["requires_api_key"] and not api_key:
            raise ValueError(
                f"API key required for provider '{provider}'. "
                "Pass it in the request or set the environment variable."
            )

        # Dispatch to provider-specific handler
        handler = getattr(self, f"_call_{provider}", None)
        if handler is None:
            raise ValueError(f"No handler for provider: {provider}")

        response_text = await handler(prompt, api_key, model)

        return {
            "response": response_text,
            "provider": provider,
            "model": model,
        }

    # ── Provider Handlers ───────────────────────────────────────────────

    async def _call_openai(self, prompt: str, api_key: str, model: str) -> str:
        url = self.PROVIDER_CONFIGS["openai"]["base_url"]
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a dating profile generator. Always respond with valid JSON."},
                {"role": "user", "content": prompt},
            ],
            "temperature": settings.llm_temperature,
            "max_tokens": settings.llm_max_tokens,
            "response_format": {"type": "json_object"},
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    async def _call_gemini(self, prompt: str, api_key: str, model: str) -> str:
        url = self.PROVIDER_CONFIGS["gemini"]["base_url"].format(model=model)
        url = f"{url}?key={api_key}"

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": f"You are a dating profile generator. Always respond with valid JSON.\n\n{prompt}"}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": settings.llm_temperature,
                "maxOutputTokens": settings.llm_max_tokens,
                "responseMimeType": "application/json",
            },
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]

    async def _call_anthropic(self, prompt: str, api_key: str, model: str) -> str:
        url = self.PROVIDER_CONFIGS["anthropic"]["base_url"]
        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": model,
            "max_tokens": settings.llm_max_tokens,
            "system": "You are a dating profile generator. Always respond with valid JSON.",
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": settings.llm_temperature,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["content"][0]["text"]

    async def _call_ollama(self, prompt: str, api_key: str, model: str) -> str:
        base = settings.ollama_base_url.rstrip("/")
        url = f"{base}/api/chat"

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a dating profile generator. Always respond with valid JSON."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": settings.llm_temperature,
                "num_predict": settings.llm_max_tokens,
            },
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"]

    async def _call_openrouter(self, prompt: str, api_key: str, model: str) -> str:
        url = self.PROVIDER_CONFIGS["openrouter"]["base_url"]
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://voice-onboarding.app",
            "X-Title": "AI Voice Onboarding",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a dating profile generator. Always respond with valid JSON."},
                {"role": "user", "content": prompt},
            ],
            "temperature": settings.llm_temperature,
            "max_tokens": settings.llm_max_tokens,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    async def _call_minimax(self, prompt: str, api_key: str, model: str) -> str:
        url = self.PROVIDER_CONFIGS["minimax"]["base_url"]
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a dating profile generator. Always respond with valid JSON."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": settings.llm_max_tokens,
            "temperature": settings.llm_temperature,
            "top_p": 0.95,
            "stream": False,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _get_default_key(provider: str) -> Optional[str]:
        key_map = {
            "openai": settings.openai_api_key,
            "gemini": settings.gemini_api_key,
            "anthropic": settings.anthropic_api_key,
            "openrouter": settings.openrouter_api_key,
            "ollama": None,
        }
        return key_map.get(provider)

    @classmethod
    def get_providers_info(cls) -> list[dict]:
        return [
            {
                "name": name,
                "requires_api_key": config["requires_api_key"],
                "default_model": config["default_model"],
            }
            for name, config in cls.PROVIDER_CONFIGS.items()
        ]

    # ── Fallback ────────────────────────────────────────────────────────

    async def _fallback(self, input_data: dict[str, Any], error: Exception) -> dict[str, Any]:
        """Generate a mock LLM response for testing."""
        self.logger.warning(
            f"LLM call failed – generating mock response: {error}",
            extra={"service": self.name},
        )
        mock_response = json.dumps({
            "personality_summary": "A warm, adventurous soul who brings enthusiasm to every conversation. Natural storyteller with a genuine curiosity about the world.",
            "dating_bio": "Adventure seeker with a passion for cooking and a weakness for sci-fi movies. Looking for someone to explore both new cuisines and new horizons with. Equally happy on a hiking trail or trying a new restaurant downtown.",
            "compatibility_features": [
                "Shared love of outdoor activities",
                "Appreciation for good food and cooking",
                "Open to new experiences",
                "Values humor and kindness",
            ],
            "ice_breakers": [
                "What's the most adventurous meal you've ever cooked?",
                "If you could teleport anywhere right now, where would you go?",
                "What's your go-to comfort movie?",
            ],
            "green_flags": [
                "Expressive and warm communication style",
                "Clear about what they're looking for",
                "Diverse range of interests",
                "Positive and enthusiastic outlook",
            ],
            "red_flags": [
                "May prioritize adventure over stability at times",
                "Could be overly idealistic about relationships",
            ],
            "conversation_style": "Engaging and animated, with a natural flow between topics. Uses humor to connect and shows genuine interest in getting to know others.",
            "match_recommendations": [
                "Someone who values experiences over material things",
                "A fellow foodie who enjoys both cooking and dining out",
                "An adventurous spirit open to spontaneous plans",
                "Someone with a good sense of humor and emotional intelligence",
            ],
        })

        return {
            "response": mock_response,
            "provider": "mock",
            "model": "fallback",
            "fallback_reason": str(error),
        }
