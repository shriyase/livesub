"""ASR provider abstraction. Swap implementations via ASR_PROVIDER env var."""
from __future__ import annotations

import asyncio
import io
import os
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


# Codes OpenAI's gpt-4o-transcribe rejects with
# "Language code 'X' is not recognized". For these we go straight to the
# `prompt` fallback. Discovered codes are added at runtime as we see them.
_UNSUPPORTED_LANG_CODES: set[str] = {"si"}

# Full names used in the `prompt` fallback ("The following audio is in X.")
_LANG_NAMES_FOR_PROMPT: dict[str, str] = {
    "si": "Sinhala",
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
}


@dataclass
class ASRResult:
    text: str
    is_final: bool
    language: Optional[str] = None
    error: Optional[str] = None  # surfaced to client for debugging


class ASRProvider(ABC):
    """Abstract base. Implementations receive raw PCM16 mono audio and return text."""

    @abstractmethod
    async def transcribe(
        self,
        pcm16: bytes,
        sample_rate: int,
        source_language: Optional[str],
        is_final: bool,
    ) -> ASRResult:
        ...


def _pcm16_to_wav_bytes(pcm16: bytes, sample_rate: int) -> bytes:
    """Wrap raw PCM16 mono into an in-memory WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16)
    return buf.getvalue()


class OpenAIASR(ASRProvider):
    """Uses OpenAI's transcription endpoint. Default model: gpt-4o-transcribe.

    Note: as of writing, OpenAI's transcription endpoint is request/response
    (not bidi-streaming over WebSocket from Python SDK). We approximate
    "streaming" by sending small audio windows from the backend. For the
    realtime API (truly bidi), see commented hook below.
    """

    def __init__(self, model: str, api_key: str):
        from openai import AsyncOpenAI

        self.model = model
        self.client = AsyncOpenAI(api_key=api_key)

    async def _call(
        self,
        wav_bytes: bytes,
        language: Optional[str],
        prompt: Optional[str] = None,
    ) -> tuple[str, Optional[str]]:
        """Single API call. Returns (text, error_message). The SDK consumes
        the file-like object, so we re-create one per call."""
        file_tuple = ("chunk.wav", io.BytesIO(wav_bytes), "audio/wav")
        kwargs = {"model": self.model, "file": file_tuple}
        if language:
            kwargs["language"] = language
        if prompt:
            kwargs["prompt"] = prompt
        try:
            resp = await self.client.audio.transcriptions.create(**kwargs)
            return (resp.text or "").strip(), None
        except Exception as e:
            return "", f"{type(e).__name__}: {e}"

    async def transcribe(
        self,
        pcm16: bytes,
        sample_rate: int,
        source_language: Optional[str],
        is_final: bool,
    ) -> ASRResult:
        if len(pcm16) == 0:
            return ASRResult(text="", is_final=is_final, language=source_language)

        wav_bytes = _pcm16_to_wav_bytes(pcm16, sample_rate)
        lang_hint: Optional[str] = None
        # OpenAI expects ISO 639-1 like "es", "fr". We pass the prefix.
        if source_language and source_language != "auto":
            lang_hint = source_language.split("-")[0]

        # Skip the language= call entirely for codes we know gpt-4o-transcribe
        # rejects ("Language code 'X' is not recognized"). For these, jump
        # straight to the prompt-based hint so we don't waste a round-trip.
        if lang_hint and lang_hint in _UNSUPPORTED_LANG_CODES:
            return await self._transcribe_with_prompt(
                wav_bytes, lang_hint, source_language, is_final
            )

        # First attempt: with language hint (faster + more accurate when supported)
        text, err = await self._call(wav_bytes, lang_hint)
        if text:
            return ASRResult(text=text, is_final=is_final, language=source_language)

        # If the language hint caused the failure, retry using the `prompt`
        # field instead. OpenAI's own error message recommends this:
        # "Try adding the language name to your prompt."
        if lang_hint and err and "not recognized" in err.lower():
            _UNSUPPORTED_LANG_CODES.add(lang_hint)  # cache so we skip next time
            print(
                f"[OpenAIASR] language={lang_hint} not accepted; "
                f"retrying with prompt-based hint"
            )
            return await self._transcribe_with_prompt(
                wav_bytes, lang_hint, source_language, is_final
            )

        # Other errors: return as-is so the WS layer can surface them
        if err:
            print(f"[OpenAIASR] error: {err}")
            return ASRResult(
                text="", is_final=is_final, language=source_language, error=err,
            )
        return ASRResult(text="", is_final=is_final, language=source_language)

    async def _transcribe_with_prompt(
        self,
        wav_bytes: bytes,
        lang_code: str,
        source_language: Optional[str],
        is_final: bool,
    ) -> ASRResult:
        """Transcribe without a `language` parameter, biasing the model
        toward the desired language via the `prompt` field instead."""
        lang_name = _LANG_NAMES_FOR_PROMPT.get(lang_code, lang_code)
        prompt = f"The following audio is in {lang_name}."
        text, err = await self._call(wav_bytes, language=None, prompt=prompt)
        if text:
            return ASRResult(
                text=text, is_final=is_final, language=source_language,
            )
        if err:
            print(f"[OpenAIASR] prompt-fallback error: {err}")
        return ASRResult(
            text="", is_final=is_final, language=source_language, error=err,
        )


class FasterWhisperASR(ASRProvider):
    """Local ASR fallback using faster-whisper. Requires `pip install faster-whisper`."""

    def __init__(self, model_size: str = "small", device: str = "cpu"):
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise RuntimeError(
                "faster-whisper not installed. `pip install faster-whisper` "
                "or set ASR_PROVIDER=openai"
            ) from e

        self.model = WhisperModel(model_size, device=device, compute_type="int8")

    async def transcribe(
        self,
        pcm16: bytes,
        sample_rate: int,
        source_language: Optional[str],
        is_final: bool,
    ) -> ASRResult:
        if len(pcm16) == 0:
            return ASRResult(text="", is_final=is_final, language=source_language)

        audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        lang = None
        if source_language and source_language != "auto":
            lang = source_language.split("-")[0]

        # faster-whisper is sync; for MVP we run it inline. Move to a thread
        # pool (asyncio.to_thread) if it becomes a bottleneck.
        segments, info = self.model.transcribe(
            audio,
            language=lang,
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        text = "".join(seg.text for seg in segments).strip()
        return ASRResult(text=text, is_final=is_final, language=info.language)


class HuggingFaceWhisperASR(ASRProvider):
    """Loads a fine-tuned Whisper checkpoint from the Hugging Face Hub
    (or a local path) via the transformers pipeline.

    Used for languages where the OpenAI-hosted models perform poorly or
    reject the language code outright — Sinhala being the canonical case.
    Defaults to the `seniruk/whisper-small-si` community fine-tune.

    Trade-off vs. faster-whisper: transformers loads the original HF
    checkpoint format with no conversion step, but uses more RAM and is
    slower per inference. Adequate for single-user, single-language MVP.
    For lower latency you'd convert the checkpoint to CTranslate2 and
    use FasterWhisperASR pointed at the local path.
    """

    def __init__(
        self,
        model_id: str,
        language_code: str,
        device: Optional[str] = None,
    ):
        try:
            from transformers import pipeline
            import torch
        except ImportError as e:
            raise RuntimeError(
                "transformers + torch not installed. Run "
                "`pip install transformers torch` to use the HF Whisper provider, "
                "or unset SINHALA_ASR_MODEL to disable it."
            ) from e

        # Pick a device. CPU is the safe default — Apple Silicon CPU runs
        # whisper-small at ~1x realtime, which is fine for single-user.
        # Set HF_DEVICE=mps or HF_DEVICE=cuda:0 to override.
        if device is None:
            device = os.getenv("HF_DEVICE", "cpu")

        self.language_code = language_code
        self.model_id = model_id

        print(
            f"[HuggingFaceWhisperASR] loading {model_id} on device={device} "
            f"(first run will download weights, ~250-500MB)..."
        )
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=device,
        )
        print(f"[HuggingFaceWhisperASR] {model_id} ready.")

    async def transcribe(
        self,
        pcm16: bytes,
        sample_rate: int,
        source_language: Optional[str],
        is_final: bool,
    ) -> ASRResult:
        if len(pcm16) == 0:
            return ASRResult(
                text="", is_final=is_final, language=self.language_code
            )

        audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0

        # Short-form decoding with deterministic sampling. Our audio buffers
        # are short (<8s), so we don't need long-form chunked decoding —
        # which would trigger Whisper's temperature-fallback loop on any
        # threshold miss and produce worse output at higher temps.
        # Hallucination defense lives upstream (the has_speech gate) and
        # downstream (the _is_hallucination filter in main.py).
        try:
            result = await asyncio.to_thread(
                self.pipe,
                {"sampling_rate": sample_rate, "raw": audio},
                generate_kwargs={
                    "task": "transcribe",
                    "language": self.language_code,
                    "temperature": 0.0,
                    "num_beams": 5,  # Whisper's standard beam size
                    "do_sample": False,
                },
            )
            text = (result.get("text", "") or "").strip()
            return ASRResult(
                text=text, is_final=is_final, language=self.language_code,
            )
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            print(f"[HuggingFaceWhisperASR] error: {err}")
            return ASRResult(
                text="", is_final=is_final, language=self.language_code, error=err,
            )


class Wav2Vec2BertASR(ASRProvider):
    """Loads a fine-tuned Wav2Vec2-BERT (SeamlessM4T family) checkpoint.

    Used specifically for Sinhala via the L-Inuri/Wav2Vec-BERT model from
    Weerakoon et al. (UCSC, 2025), which reports 1.79% WER on a 40-hour
    Sinhala dataset — order-of-magnitude better than any Whisper Sinhala
    fine-tune currently available.

    Wav2Vec2-BERT is a CTC-based architecture, not encoder-decoder like
    Whisper, so we don't pass language/task kwargs. The transformers
    pipeline auto-detects the architecture and dispatches correctly.
    """

    def __init__(
        self,
        model_id: str,
        language_code: str,
        device: Optional[str] = None,
    ):
        try:
            from transformers import pipeline
            import torch  # noqa: F401  (used for device detection)
        except ImportError as e:
            raise RuntimeError(
                "transformers + torch not installed. Run "
                "`pip install transformers torch` to use Wav2Vec2BertASR."
            ) from e

        if device is None:
            device = os.getenv("HF_DEVICE", "cpu")

        self.language_code = language_code
        self.model_id = model_id

        print(
            f"[Wav2Vec2BertASR] loading {model_id} on device={device} "
            f"(first run will download weights, ~600MB-2GB)..."
        )
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=device,
        )
        print(f"[Wav2Vec2BertASR] {model_id} ready.")

    async def transcribe(
        self,
        pcm16: bytes,
        sample_rate: int,
        source_language: Optional[str],
        is_final: bool,
    ) -> ASRResult:
        if len(pcm16) == 0:
            return ASRResult(
                text="", is_final=is_final, language=self.language_code
            )

        audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0

        # Wav2Vec2-BERT is a CTC model — no language/task kwargs, no beam
        # search by default. The pipeline handles tokenization + decoding.
        try:
            result = await asyncio.to_thread(
                self.pipe,
                {"sampling_rate": sample_rate, "raw": audio},
            )
            text = (result.get("text", "") or "").strip()
            return ASRResult(
                text=text, is_final=is_final, language=self.language_code,
            )
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            print(f"[Wav2Vec2BertASR] error: {err}")
            return ASRResult(
                text="", is_final=is_final, language=self.language_code, error=err,
            )


class RoutingASR(ASRProvider):
    """Per-language routing wrapper. Calls the override provider for any
    language code in `overrides`, otherwise falls through to `default`.

    This keeps OpenAI as the primary path for the 95% case while letting
    us special-case low-resource languages like Sinhala that need a
    fine-tuned local model.
    """

    def __init__(
        self,
        default: ASRProvider,
        overrides: dict[str, ASRProvider],
    ):
        self.default = default
        self.overrides = overrides

    async def transcribe(
        self,
        pcm16: bytes,
        sample_rate: int,
        source_language: Optional[str],
        is_final: bool,
    ) -> ASRResult:
        provider = self.default
        if source_language and source_language != "auto":
            lang_code = source_language.split("-")[0]
            if lang_code in self.overrides:
                provider = self.overrides[lang_code]
        return await provider.transcribe(
            pcm16, sample_rate, source_language, is_final
        )


def _build_primary_provider() -> ASRProvider:
    provider = os.getenv("ASR_PROVIDER", "openai").lower()
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY missing")
        model = os.getenv("OPENAI_ASR_MODEL", "gpt-4o-transcribe")
        return OpenAIASR(model=model, api_key=api_key)
    if provider == "faster_whisper":
        return FasterWhisperASR()
    raise ValueError(f"Unknown ASR_PROVIDER: {provider}")


def build_asr_provider() -> ASRProvider:
    """Build the active ASR provider, including any per-language overrides.

    Sinhala routing: set SINHALA_ASR_MODEL to a Hugging Face repo id (or a
    local path) to route Sinhala audio to a fine-tuned local model instead
    of OpenAI. SINHALA_ASR_KIND selects the architecture:

      - "wav2vec2bert"  → Wav2Vec2-BERT (recommended, ~1.79% WER with
                          L-Inuri/Wav2Vec-BERT from Weerakoon et al. 2025)
      - "whisper"       → Whisper-family fine-tune (e.g. seniruk/whisper-small-si)

    Default kind is auto-detected from the model id (anything containing
    "wav2vec" → wav2vec2bert; otherwise whisper).
    """
    primary = _build_primary_provider()

    overrides: dict[str, ASRProvider] = {}

    sinhala_model = os.getenv("SINHALA_ASR_MODEL", "").strip()
    if sinhala_model:
        kind = os.getenv("SINHALA_ASR_KIND", "").strip().lower()
        if not kind:
            kind = "wav2vec2bert" if "wav2vec" in sinhala_model.lower() else "whisper"
        try:
            if kind == "wav2vec2bert":
                overrides["si"] = Wav2Vec2BertASR(
                    model_id=sinhala_model, language_code="si"
                )
            elif kind == "whisper":
                overrides["si"] = HuggingFaceWhisperASR(
                    model_id=sinhala_model, language_code="si"
                )
            else:
                raise ValueError(f"Unknown SINHALA_ASR_KIND: {kind}")
        except Exception as e:
            # Don't crash the whole backend if HF/transformers can't load —
            # fall back to OpenAI for Sinhala (with the prompt-based hint).
            print(
                f"[build_asr_provider] failed to load Sinhala model "
                f"({sinhala_model}, kind={kind}): {e}\n"
                f"  Falling back to primary provider for Sinhala."
            )

    if overrides:
        return RoutingASR(default=primary, overrides=overrides)
    return primary
