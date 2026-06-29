# LiveSub

Real-time translated subtitles from live microphone audio. Speak in any of 10+ languages, see English subtitles updating with 1–3 second latency.

## Architecture

```
mic → 16kHz PCM16 chunks → WebSocket → FastAPI
   → buffer + energy VAD
   → ASRProvider (OpenAI gpt-4o-transcribe | faster-whisper local)
   → SubtitleStabilizer
   → TranslationProvider (gpt-4o-mini)
   → WebSocket JSON ({type: partial|final, ...})
   → React UI: live revisable line + finalized history
```

Both ASR and Translation are behind abstract base classes (`providers/asr.py`, `providers/translation.py`) — drop in another implementation and select it via `ASR_PROVIDER` / `TRANSLATION_PROVIDER`.

## Setup

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
python main.py
```

Backend listens on `http://localhost:8000`. WebSocket endpoint: `ws://localhost:8000/ws/audio`. Health: `GET /health`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. Allow microphone access, pick a source language, hit **Start**.

If your backend runs elsewhere, set `VITE_BACKEND_WS` before `npm run dev`:

```bash
VITE_BACKEND_WS=ws://192.168.1.10:8000/ws/audio npm run dev
```

## Test script

Stream a pre-recorded WAV (mono, 16-bit, 16kHz) to the backend and see the subtitle stream printed:

```bash
python test_client.py samples/spanish_clip.wav --source es --target en
```

To convert a clip to the right format:

```bash
ffmpeg -i input.mp3 -ac 1 -ar 16000 -sample_fmt s16 spanish_clip.wav
```

## Supported source languages

Auto-detect (beta), Sinhala, Spanish, French, German, Hindi, Arabic, Mandarin Chinese, Tamil, Japanese, Korean. Target is English.

Auto-detect is off by default because the model can flip languages between utterances when the speaker pauses. Pick the source language manually for stable results.

## Swapping providers

**Use local Whisper instead of OpenAI for ASR:**

```bash
pip install faster-whisper
# in .env
ASR_PROVIDER=faster_whisper
```

**Add a new provider** — subclass `ASRProvider` (or `TranslationProvider`), implement `transcribe` (or `translate`), and add it to the `build_*_provider` factory.

## How real-time works

- Frontend captures mic at the device's native rate, downsamples to 16kHz mono PCM16, and pushes ~85ms chunks over the WebSocket.
- Backend appends to a rolling buffer. Once the buffer exceeds `MIN_PARTIAL_MS` (600ms), it kicks off an ASR + translation pass and sends a `partial` message. Only one partial is in flight at a time — newer audio supersedes older requests.
- An energy-based VAD watches the trailing `VAD_SILENCE_MS` (600ms). When silence is detected, the buffer is finalized: a `final` message is emitted and the buffer is cleared.
- Buffer is also force-finalized at `MAX_BUFFER_MS` (8s) to bound latency for monologues.
- Frontend shows the latest `partial` as a revisable "live" line and appends each `final` to the history list.

## MVP non-goals

No speaker diarization, no offline mode, no audio persistence, no auth, no voice output. The translation prompt is intentionally minimal — for production you'd want context windows across recent finals, glossary handling, and per-language polish.

## Privacy

Audio is processed in-memory only. The backend never writes raw audio to disk and never logs PCM bytes. If you add opt-in persistence, do it explicitly — there's a comment marker in `backend/main.py` (`PRIVACY:`) where the consent gate belongs.

## Known MVP limitations

- The OpenAI transcription endpoint is request/response, not bidi-streaming, so each partial is a fresh upload of the rolling buffer. Latency is dominated by buffer size + round-trip. For sub-second latency, swap to the OpenAI Realtime API or Deepgram streaming — the `ASRProvider` interface already supports it; you'd implement a long-lived session inside the provider rather than per-call uploads.
- Energy-based VAD will mis-trigger in noisy rooms. Swap to Silero VAD or WebRTC VAD for production.
- `ScriptProcessorNode` is deprecated in browsers. Works today; migrate to `AudioWorklet` for production.
