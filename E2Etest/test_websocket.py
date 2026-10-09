"""Stream a local wav to /realtime/stream as PCM s16le.

    conda activate lingting
    python E2Etest/test_websocket.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from math import gcd
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import websockets

# --- 在这里改录音文件 ---
WAV_PATH = r"C:\Users\Numb\Desktop\11s.wav"
LANGUAGE = "auto"  # zh / en / auto
BASE_URL = "http://127.0.0.1:8765"
WS_URL = "ws://127.0.0.1:8765/realtime/stream"
USE_VAD = True
REALTIME = True
CHUNK_MS = 200
TARGET_SR = 16000


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


def _print_msg(msg: dict) -> None:
    if "text" in msg:
        kind = msg.get("type") or ("final" if msg.get("is_final") else "partial")
        extra = ""
        if kind == "final":
            extra = f" {msg.get('start')}-{msg.get('end')} speaker={msg.get('speaker')}"
        print(f"[{kind}]{extra} {msg.get('text', '')}")
        for item in msg.get("chars") or msg.get("words") or []:
            print(f"  {item.get('start'):.3f}-{item.get('end'):.3f} {item.get('text') or item.get('word')}")
        return
    print("recv:", msg)


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
    print("health:", health.json())

    pcm, sr, duration = load_pcm_s16le(wav)
    chunk_bytes = max(2, int(sr * CHUNK_MS / 1000) * 2)
    print(f"wav={wav} duration={duration:.2f}s sr={sr} bytes={len(pcm)} vad={USE_VAD}")

    url = f"{WS_URL}?sample_rate={sr}&use_server_vad={'true' if USE_VAD else 'false'}"
    async with websockets.connect(url, max_size=None, open_timeout=10) as ws:
        hello = json.loads(await ws.recv())
        print("connected:", hello)
        if hello.get("status") != "connected":
            return 1

        await ws.send(json.dumps({
            "type": "start",
            "language": LANGUAGE,
            "sample_rate": sr,
            "channels": 1,
            "format": "pcm_s16le",
            "use_server_vad": USE_VAD,
        }))
        print("configured:", json.loads(await ws.recv()))

        finals: list[str] = []
        got_final = asyncio.Event()

        async def reader():
            try:
                while True:
                    raw = await ws.recv()
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8")
                    msg = json.loads(raw)
                    print(msg)
                    _print_msg(msg)
                    if msg.get("type") == "final" or msg.get("is_final"):
                        finals.append(msg.get("text") or "")
                        got_final.set()
            except websockets.ConnectionClosed:
                got_final.set()

        reader_task = asyncio.create_task(reader())
        try:
            offset = 0
            while offset < len(pcm):
                await ws.send(pcm[offset : offset + chunk_bytes])
                offset += chunk_bytes
                if REALTIME:
                    await asyncio.sleep(CHUNK_MS / 1000.0)
            got_final.clear()
            await ws.send(json.dumps({"type": "flush"}))
            await asyncio.wait_for(got_final.wait(), timeout=120.0)
        except asyncio.TimeoutError:
            print("timeout waiting for flush final", file=sys.stderr)
        finally:
            reader_task.cancel()
            try:
                await reader_task
            except asyncio.CancelledError:
                pass

    text = "".join(finals).strip() or (finals[-1] if finals else "")
    print("final_text:", text)
    return 0


def main() -> int:
    try:
        return asyncio.run(run())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
