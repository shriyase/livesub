"""LiveSub backend.

WebSocket protocol on /ws/audio:

Client -> Server:
  - Text frame (JSON): {"type": "config", "source_language": "es",
                        "target_language": "en"}
  - Binary frames: raw PCM16 mono @ SAMPLE_RATE (configured via env)
  - Text frame (JSON): {"type": "stop"}

Server -> Client:
  - {"type": "partial" | "final",
     "source_text": "...",
     "translated_text": "...",
     "source_language": "...",
     "target_language": "en",
     "timestamp": "ISO8601"}
  - {"type": "error", "message": "..."}
"""
from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from providers import (
    SubtitleStabilizer,
    build_asr_provider,
    build_translation_provider,
)
from vad import has_speech, is_silence

load_dotenv()

SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
CHUNK_MS = int(os.getenv("CHUNK_MS", "750"))
VAD_SILENCE_MS = int(os.getenv("VAD_SILENCE_MS", "600"))
VAD_ENERGY_THRESHOLD = float(os.getenv("VAD_ENERGY_THRESHOLD", "0.005"))

# Minimum audio buffered before we attempt a partial transcription.
# Below this, ASR tends to return empty / hallucinated text.
# Whisper-small fine-tunes are unreliable below ~2s; raise via .env if
# you're using a local Whisper model and seeing flicker/gibberish on
# partials.
MIN_PARTIAL_MS = int(os.getenv("MIN_PARTIAL_MS", "600"))
# Cap how long the buffer can grow between finalizations to keep latency bounded
MAX_BUFFER_MS = int(os.getenv("MAX_BUFFER_MS", "8000"))
# Set to "1" to skip partial transcriptions entirely — only emit on
# end-of-utterance (silence) or buffer-full. Recommended when using a
# local Whisper model that performs poorly on short audio windows.
DISABLE_PARTIALS = os.getenv("DISABLE_PARTIALS", "0") == "1"
# Energy threshold used specifically to gate ASR (must be loud enough to be
# real speech, not just background hum). Slightly higher than the silence
# threshold so we err on the side of NOT calling the API on near-silence.
SPEECH_ENERGY_THRESHOLD = float(
    os.getenv("SPEECH_ENERGY_THRESHOLD", str(VAD_ENERGY_THRESHOLD * 3))
)
# How much voiced audio (above threshold) we need before bothering ASR
MIN_SPEECH_MS = int(os.getenv("MIN_SPEECH_MS", "350"))

# Common Whisper / gpt-4o-transcribe hallucinations on silence/noise.
# If the model returns one of these (case-insensitive, stripped of trailing
# punctuation), we treat it as garbage and don't emit a subtitle.
_HALLUCINATIONS = {
    "thank you", "thanks", "thank you for watching", "thanks for watching",
    "you", "yeah", "okay", "ok", "uh", "um", "hmm", "mhm",
    "welcome", "hello", "hi", "bye", "goodbye",
    "light", "everyone", "your", "the", "a", "and",
    ".", "...", "?", "!",
    "music", "[music]", "[music playing]", "(music)",
    "applause", "[applause]", "(applause)",
    "silence", "[silence]",
    "subtitles by the amara.org community",
    "subscribe", "like and subscribe",
}


def _is_hallucination(text: str) -> bool:
    if not text:
        return True
    cleaned = text.strip().strip(".?!,;:").lower()
    if cleaned in _HALLUCINATIONS or len(cleaned) <= 2:
        return True
    # Repetition: Whisper-style hallucinations on silence often produce
    # the same short token repeated. If >50% of the (whitespace-split)
    # tokens are the same, treat as a hallucination. Works for any script.
    tokens = cleaned.split()
    if len(tokens) >= 3:
        most_common_count = max(tokens.count(t) for t in set(tokens))
        if most_common_count / len(tokens) > 0.5:
            return True
    # Character-level repetition (e.g. "අඅඅඅ" or "ahahahaha")
    if len(cleaned) >= 4:
        # If the output is just one character repeated, drop it
        if len(set(cleaned.replace(" ", ""))) <= 2:
            return True
    return False

app = FastAPI(title="LiveSub")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build providers once at startup; they're stateless across connections.
asr_provider = build_asr_provider()
translation_provider = build_translation_provider()


@app.get("/health")
async def health():
    return {
        "ok": True,
        "asr_provider": os.getenv("ASR_PROVIDER", "openai"),
        "translation_provider": os.getenv("TRANSLATION_PROVIDER", "openai"),
        "sample_rate": SAMPLE_RATE,
    }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _send_subtitle(
    ws: WebSocket,
    msg_type: str,
    source_text: str,
    translated_text: str,
    source_language: str,
    target_language: str,
):
    payload = {
        "type": msg_type,
        "source_text": source_text,
        "translated_text": translated_text,
        "source_language": source_language,
        "target_language": target_language,
        "timestamp": _now_iso(),
    }
    await ws.send_text(json.dumps(payload))


@app.websocket("/ws/audio")
async def ws_audio(ws: WebSocket):
    await ws.accept()
    stabilizer = SubtitleStabilizer()
    source_language = "auto"
    target_language = "en"

    # Single in-flight ASR/translation pass — we drop overlapping requests
    # rather than queue them, so subtitles always reflect recent audio.
    in_flight: asyncio.Task | None = None
    last_partial_at_ms = 0.0

    # PRIVACY: raw audio is never written to disk or logged here.
    # If you ever add opt-in persistence, do it explicitly behind a user
    # consent flag — and only after this point.

    try:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                break

            if "text" in message and message["text"] is not None:
                try:
                    data = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue

                if data.get("type") == "config":
                    source_language = data.get("source_language", "auto") or "auto"
                    target_language = data.get("target_language", "en") or "en"
                    print(
                        f"[ws] config source={source_language} target={target_language}"
                    )
                elif data.get("type") == "stop":
                    # Force-finalize whatever is buffered
                    await _finalize_now(
                        ws, stabilizer, source_language, target_language
                    )
                    break

            elif "bytes" in message and message["bytes"] is not None:
                stabilizer.append_audio(message["bytes"])
                buf_ms = stabilizer.buffer_duration_ms(SAMPLE_RATE)

                # Detect end-of-utterance via silence in the trailing window
                trailing_silence = is_silence(
                    stabilizer.buffer_bytes(),
                    SAMPLE_RATE,
                    VAD_SILENCE_MS,
                    VAD_ENERGY_THRESHOLD,
                )

                should_finalize = (
                    trailing_silence and buf_ms > MIN_PARTIAL_MS
                ) or buf_ms >= MAX_BUFFER_MS

                if should_finalize:
                    if in_flight and not in_flight.done():
                        in_flight.cancel()
                    await _finalize_now(
                        ws, stabilizer, source_language, target_language
                    )
                    last_partial_at_ms = 0.0
                    continue

                # Throttle partials: at most one in flight, and at least
                # CHUNK_MS between kickoffs. Skip entirely if disabled.
                if (
                    not DISABLE_PARTIALS
                    and buf_ms >= MIN_PARTIAL_MS
                    and (buf_ms - last_partial_at_ms) >= CHUNK_MS
                    and (in_flight is None or in_flight.done())
                ):
                    last_partial_at_ms = buf_ms
                    in_flight = asyncio.create_task(
                        _emit_partial(
                            ws,
                            stabilizer,
                            source_language,
                            target_language,
                        )
                    )

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass
    finally:
        if in_flight and not in_flight.done():
            in_flight.cancel()
        try:
            await ws.close()
        except Exception:
            pass


async def _emit_partial(
    ws: WebSocket,
    stabilizer: SubtitleStabilizer,
    source_language: str,
    target_language: str,
):
    audio = stabilizer.buffer_bytes()
    # Gate: skip ASR entirely if the buffer doesn't contain real speech.
    # This is the main defense against the model hallucinating words on
    # silence or background noise.
    if not has_speech(audio, SAMPLE_RATE, SPEECH_ENERGY_THRESHOLD, MIN_SPEECH_MS):
        return

    asr_lang = None if source_language == "auto" else source_language
    asr_result = await asr_provider.transcribe(
        audio, SAMPLE_RATE, asr_lang, is_final=False
    )
    text = asr_result.text.strip()
    if not text:
        # Surface ASR errors (e.g. unsupported language) once, so the user
        # isn't staring at a silent UI wondering what's wrong.
        if asr_result.error:
            await ws.send_text(json.dumps({
                "type": "error",
                "message": f"ASR: {asr_result.error}",
            }))
        return
    if _is_hallucination(text):
        return

    translated = await translation_provider.translate(
        text, asr_result.language or source_language, target_language
    )
    if not translated:
        return

    stabilizer.update_partial(text, translated)
    await _send_subtitle(
        ws,
        "partial",
        text,
        translated,
        asr_result.language or source_language,
        target_language,
    )


async def _finalize_now(
    ws: WebSocket,
    stabilizer: SubtitleStabilizer,
    source_language: str,
    target_language: str,
):
    audio = stabilizer.buffer_bytes()
    if len(audio) == 0:
        return

    # Same gate as in _emit_partial — don't transcribe pure silence/noise,
    # the model will hallucinate ("Welcome.", "Thank you.", etc).
    if not has_speech(audio, SAMPLE_RATE, SPEECH_ENERGY_THRESHOLD, MIN_SPEECH_MS):
        stabilizer.audio_buffer.clear()
        return

    asr_lang = None if source_language == "auto" else source_language
    asr_result = await asr_provider.transcribe(
        audio, SAMPLE_RATE, asr_lang, is_final=True
    )
    text = asr_result.text.strip()
    if not text or _is_hallucination(text):
        stabilizer.audio_buffer.clear()
        return

    translated = await translation_provider.translate(
        text, asr_result.language or source_language, target_language
    )
    stabilizer.finalize(text, translated)
    await _send_subtitle(
        ws,
        "final",
        text,
        translated,
        asr_result.language or source_language,
        target_language,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
