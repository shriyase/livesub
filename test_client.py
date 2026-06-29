"""Simple end-to-end test: streams a WAV file to the backend WebSocket
and prints translated subtitles as they arrive.

Usage:
    python test_client.py path/to/audio.wav --source es --target en
"""
from __future__ import annotations

import argparse
import asyncio
import json
import wave

import websockets

SAMPLE_RATE = 16000
CHUNK_MS = 250  # how often we push audio (simulates real-time mic)


async def stream_wav(path: str, source: str, target: str, url: str):
    with wave.open(path, "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != SAMPLE_RATE:
            raise SystemExit(
                f"Expected mono 16-bit {SAMPLE_RATE}Hz WAV; got "
                f"{wf.getnchannels()}ch / {wf.getsampwidth()*8}bit / {wf.getframerate()}Hz"
            )

        frames_per_chunk = int(SAMPLE_RATE * (CHUNK_MS / 1000.0))
        chunks = []
        while True:
            data = wf.readframes(frames_per_chunk)
            if not data:
                break
            chunks.append(data)

    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({
            "type": "config",
            "source_language": source,
            "target_language": target,
        }))

        async def receiver():
            try:
                async for msg in ws:
                    data = json.loads(msg)
                    tag = data.get("type", "?").upper()
                    print(
                        f"[{tag:7}] {data.get('translated_text','')}  "
                        f"<<{data.get('source_text','')}>>"
                    )
            except websockets.ConnectionClosed:
                pass

        recv_task = asyncio.create_task(receiver())

        for c in chunks:
            await ws.send(c)
            await asyncio.sleep(CHUNK_MS / 1000.0)

        await ws.send(json.dumps({"type": "stop"}))

        try:
            await asyncio.wait_for(recv_task, timeout=10)
        except asyncio.TimeoutError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--source", default="es")
    ap.add_argument("--target", default="en")
    ap.add_argument("--url", default="ws://localhost:8000/ws/audio")
    args = ap.parse_args()
    asyncio.run(stream_wav(args.wav, args.source, args.target, args.url))


if __name__ == "__main__":
    main()
