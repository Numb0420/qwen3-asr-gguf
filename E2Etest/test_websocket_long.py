"""Loop a local wav to /realtime/stream as PCM s16le for long-session stress test.

    conda activate lingting
    python E2Etest\test_websocket_long.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from math import gcd
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import websockets

# --- config ---
WAV_PATH = r"C:\Users\Numb\Desktop\11s.wav"
LANGUAGE = "auto"
BASE_URL = "http://127.0.0.1:8765"
WS_URL = "ws://127.0.0.1:8765/realtime/stream"
USE_VAD = True
REALTIME = True
CHUNK_MS = 200
TARGET_SR = 16000
TARGET_MINUTES = 35  # run at least this long (past the 23-min crash point)


def load_pcm_s16le(path: Path) -> tuple[bytes, int, float]:
    audio, sr = sf.read(str(path), always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if int(sr) != TARGET_SR:
        g = gcd(int(sr), TARGET_SR)
        audio = resample_poly(audio, TARGET_SR // g, int(sr) // g).astype(np.float32)
        sr = TARGET_SR
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16).tobytes()
    duration = len(audio) / float(sr)
    return pcm, int(sr), duration


async def run() -> int:
    wav = Path(WAV_PATH).expanduser()
    if not wav.is_file():
        print(f"wav not found: {wav}", file=sys.stderr)
        return 1

    try:
        health = httpx.get(f"{BASE_URL}/health", timeout=5.0)
        health.raise_for_status()
    except Exception as exc:
        print(f"server unreachable at {BASE_URL}: {exc}", file=sys.stderr)
        return 1
    print(f"health: {health.json()}")

    pcm, sr, duration = load_pcm_s16le(wav)
    chunk_bytes = max(2, int(sr * CHUNK_MS / 1000) * 2)
    print(f"wav={wav} duration={duration:.2f}s sr={sr} bytes={len(pcm)} vad={USE_VAD}")
    print(f"target: {TARGET_MINUTES} minutes of looping audio")

    url = f"{WS_URL}?sample_rate={sr}&use_server_vad={'true' if USE_VAD else 'false'}"

    start_time = time.monotonic()
    loop_count = 0
    total_finals = 0

    async with websockets.connect(url, max_size=None, open_timeout=10) as ws:
        hello = json.loads(await ws.recv())
        print(f"connected: {hello}")
        if hello.get("status") != "connected":
            return 1

        await ws.send(json.dumps({"action": "config", "language": LANGUAGE, "use_server_vad": USE_VAD}))
        print(f"configured: {json.loads(await ws.recv())}")

        got_final = asyncio.Event()
        stop_flag = asyncio.Event()

        async def reader():
            nonlocal total_finals
            try:
                while True:
                    raw = await ws.recv()
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8")
                    msg = json.loads(raw)
                    if msg.get("type") == "final" or msg.get("is_final"):
                        total_finals += 1
                        text = msg.get("text", "")
                        elapsed = time.monotonic() - start_time
                        print(f"[{elapsed:.0f}s] final #{total_finals}: {text[:80]}")
                        got_final.set()
                    # silently consume partials to avoid log spam
            except websockets.ConnectionClosed:
                print("WS connection closed by server!")
                got_final.set()
                stop_flag.set()

        reader_task = asyncio.create_task(reader())
        try:
            while True:
                elapsed = time.monotonic() - start_time
                if elapsed >= TARGET_MINUTES * 60:
                    print(f"reached target {TARGET_MINUTES} min, stopping")
                    break
                if stop_flag.is_set():
                    print("stop flag set, aborting")
                    break

                loop_count += 1
                if loop_count % 10 == 0:
                    print(f"[{elapsed:.0f}s] loop {loop_count}, finals={total_finals}")

                # send one full audio loop
                offset = 0
                while offset < len(pcm):
                    if stop_flag.is_set():
                        break
                    await ws.send(pcm[offset : offset + chunk_bytes])
                    offset += chunk_bytes
                    if REALTIME:
                        await asyncio.sleep(CHUNK_MS / 1000.0)

                # flush after each loop
                got_final.clear()
                await ws.send(json.dumps({"action": "flush"}))
                try:
                    await asyncio.wait_for(got_final.wait(), timeout=30.0)
                except asyncio.TimeoutError:
                    print(f"[{time.monotonic()-start_time:.0f}s] timeout waiting for flush final")

                # reset for next loop (clears KV cache to simulate new utterance)
                await ws.send(json.dumps({"action": "reset"}))

        finally:
            reader_task.cancel()
            try:
                await reader_task
            except asyncio.CancelledError:
                pass

    elapsed = time.monotonic() - start_time
    print(f"done: {elapsed:.0f}s elapsed, {loop_count} loops, {total_finals} finals")
    return 0


def main() -> int:
    try:
        return asyncio.run(run())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
