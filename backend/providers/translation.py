"""Translation provider abstraction."""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional


class TranslationProvider(ABC):
    @abstractmethod
    async def translate(
        self,
        text: str,
        source_language: Optional[str],
        target_language: str,
    ) -> str:
        ...


class OpenAITranslation(TranslationProvider):
    """Uses GPT to translate. Cheap + fast with gpt-4o-mini."""

    SYSTEM_PROMPT = (
        "You are a real-time subtitle translator. "
        "Translate the user's text to {target} accurately and naturally. "
        "Preserve meaning and tone. Do NOT add commentary, quotes, or "
        "explanations. Output only the translation. If the input is already "
        "in {target}, return it unchanged. If the input is empty or just "
        "filler sounds, return an empty string."
    )

    def __init__(self, model: str, api_key: str):
        from openai import AsyncOpenAI

        self.model = model
        self.client = AsyncOpenAI(api_key=api_key)

    async def translate(
        self,
        text: str,
        source_language: Optional[str],
        target_language: str,
    ) -> str:
        text = (text or "").strip()
        if not text:
            return ""

        target_name = _lang_name(target_language)
        system = self.SYSTEM_PROMPT.format(target=target_name)

        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": text},
                ],
                temperature=0.2,
                max_tokens=400,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"[OpenAITranslation] error: {e}")
            return ""


_LANG_NAMES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "hi": "Hindi",
    "mr": "Marathi",
    "kn": "Kannada",
    "ta": "Tamil",
    "ar": "Arabic",
    "zh": "Mandarin Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "si": "Sinhala",
}


def _lang_name(code: str) -> str:
    return _LANG_NAMES.get(code.split("-")[0], code)


def build_translation_provider() -> TranslationProvider:
    provider = os.getenv("TRANSLATION_PROVIDER", "openai").lower()
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY missing")
        model = os.getenv("OPENAI_TRANSLATION_MODEL", "gpt-4o-mini")
        return OpenAITranslation(model=model, api_key=api_key)
    raise ValueError(f"Unknown TRANSLATION_PROVIDER: {provider}")
