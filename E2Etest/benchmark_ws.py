"""Replay one WAV at real-time pace against three WebSocket partial cadences.

Run with the project's inference environment, for example:
    D:\\anaconda3\\envs\\lingting\\python.exe E2Etest\\benchmark_ws.py \
        --wav C:\\Users\\Numb\\Desktop\\audio.wav

Each case starts its own server with the same .env and model. Only the partial
cadence changes. The final transcript is saved for inspection, not scored as
accuracy unless an independent reference transcript is supplied.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
import websockets

from test_websocket import load_pcm_s16le

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "http://127.0.0.1:8765"
WS_URL = "ws://127.0.0.1:8765/realtime/stream"
CASES = (
    ("baseline_1p2", "1.2", "0"),
    ("every_1p0", "1.0", "0"),
    ("adaptive_1p0_2p0", "1.0", "2.0"),
)
TIMING_RE = re.compile(
    r"WS timing \| kind=(\w+) .*? audio_dur=([\d.]+)s "
    r"total_ms=([\d.]+) queue_ms=([-\d.]+) infer_ms=([-\d.]+) "
    r"encode_ms=([-\d.]+) prefill_ms=([-\d.]+) decode_ms=([-\d.]+) "
    r"n_chunks=([-\d.]+) cache_hits=(\d+) decode_skipped=(\d+) result=(\w+)"
)
CLOCKS_RE = re.compile(
    r"WS clocks \| kind=(\w+) .*? text_changed=(-?\d+)"
    r"(?: .*? final_refresh_gap_s=([-\d.]+))?"
)


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct / 100.0
    lo = int(index)
    hi = min(lo + 1, len(ordered) - 1)
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo), 3)


def _server_health() -> dict | None:
    try:
        response = httpx.get(f"{BASE_URL}/health", timeout=2.0)
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, ValueError):
        return None


def start_server(case_dir: Path, partial_s: str, long_s: str) -> tuple[subprocess.Popen, object]:
    environment = os.environ.copy()
    environment.update(
        WS_PARTIAL_SECONDS=partial_s,
        LOG_DIR=str(case_dir / "logs"),
    )
    output = (case_dir / "service.out.log").open("w", encoding="utf-8")
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    process = subprocess.Popen(
        [sys.executable, str(ROOT / "run.py")],
        cwd=ROOT,
        env=environment,
        stdout=output,
        stderr=subprocess.STDOUT,
        creationflags=flags,
    )
    deadline = time.monotonic() + 90.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output.close()
            raise RuntimeError(f"server exited with {process.returncode}; see {case_dir / 'service.out.log'}")
        if _server_health() is not None:
            return process, output
        time.sleep(0.5)
    output.close()
    raise TimeoutError(f"server did not become healthy; see {case_dir / 'service.out.log'}")


def stop_server(process: subprocess.Popen, output: object) -> None:
    try:
        httpx.post(f"{BASE_URL}/admin/shutdown", timeout=5.0)
    except httpx.HTTPError:
        pass
    try:
        process.wait(timeout=20.0)
    except subprocess.TimeoutExpired:
        process.terminate()
        process.wait(timeout=10.0)
    output.close()


async def replay(
    pcm: bytes, sample_rate: int, case_dir: Path, chunk_ms: int
) -> dict:
    chunk_bytes = max(2, int(sample_rate * chunk_ms / 1000.0) * 2)
    records: list[dict] = []
    finals: list[str] = []
    sent_audio_s = 0.0
    started_at = 0.0
    url = f"{WS_URL}?sample_rate={sample_rate}&use_server_vad=true"
    async with websockets.connect(url, max_size=None, open_timeout=20, ping_interval=None) as ws:
        hello = json.loads(await ws.recv())
        if hello.get("status") != "connected":
            raise RuntimeError(f"WebSocket connection failed: {hello}")
        await ws.send(json.dumps({
            "type": "start",
            "language": "auto",
            "sample_rate": sample_rate,
            "channels": 1,
            "format": "pcm_s16le",
            "use_server_vad": True,
        }))
        configured = json.loads(await ws.recv())
        if configured.get("status") != "configured":
            raise RuntimeError(f"WebSocket configuration failed: {configured}")
        with (case_dir / "messages.jsonl").open("w", encoding="utf-8") as sink:
            async def reader() -> None:
                while True:
                    message = json.loads(await ws.recv())
                    elapsed = time.monotonic() - started_at
                    record = {
                        "received_s": round(elapsed, 3),
                        "sent_audio_s": round(sent_audio_s, 3),
                        "message": message,
                    }
                    audio_end = message.get("audio_end")
                    if isinstance(audio_end, (int, float)):
                        record["result_lag_s"] = round(elapsed - float(audio_end), 3)
                    records.append(record)
                    sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                    sink.flush()
                    if message.get("type") == "final":
                        finals.append(message.get("text") or "")
                    if message.get("status") == "stopped":
                        return

            started_at = time.monotonic()
            reader_task = asyncio.create_task(reader())
            for offset in range(0, len(pcm), chunk_bytes):
                # Absolute scheduling avoids accumulating asyncio.sleep drift.
                deadline = started_at + offset / (sample_rate * 2.0)
                delay = deadline - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                packet = pcm[offset : offset + chunk_bytes]
                await ws.send(packet)
                sent_audio_s = (offset + len(packet)) / (sample_rate * 2.0)
            await ws.send(json.dumps({"type": "stop"}))
            await asyncio.wait_for(reader_task, timeout=180.0)

    (case_dir / "final.txt").write_text("".join(finals), encoding="utf-8")
    partial_lags = [
        r["result_lag_s"] for r in records
        if r["message"].get("type") == "partial" and "result_lag_s" in r
    ]
    final_lags = [
        r["result_lag_s"] for r in records
        if r["message"].get("type") == "final"
        and not r["message"].get("boundary_tail")
        and "result_lag_s" in r
    ]
    boundary_tails = sum(bool(r["message"].get("boundary_tail")) for r in records)
    return {
        "connected": hello,
        "partial_count": len(partial_lags),
        "final_count": len(final_lags),
        "boundary_tail_count": boundary_tails,
        "partial_lag_p50_s": percentile(partial_lags, 50),
        "partial_lag_p95_s": percentile(partial_lags, 95),
        "final_lag_p50_s": percentile(final_lags, 50),
        "final_lag_p95_s": percentile(final_lags, 95),
        "replay_wall_s": round(time.monotonic() - started_at, 3),
        "sent_audio_s": round(sent_audio_s, 3),
    }


def summarize_log(case_dir: Path) -> dict:
    path = case_dir / "logs" / "server.log"
    lines = path.read_text(encoding="utf-8").splitlines()
    metrics: dict[str, list[dict]] = {"partial": [], "final": []}
    for line in lines:
        match = TIMING_RE.search(line)
        if not match:
            continue
        (
            kind, duration, total, queue, infer, encode, prefill, decode,
            n_chunks, cache_hits, decode_skipped, result,
        ) = match.groups()
        if kind in metrics:
            metrics[kind].append({
                "audio_dur": float(duration),
                "total_ms": float(total),
                "queue_ms": float(queue),
                "infer_ms": float(infer),
                "encode_ms": float(encode),
                "prefill_ms": float(prefill),
                "decode_ms": float(decode),
                "n_chunks": float(n_chunks),
                "cache_hits": float(cache_hits),
                "decode_skipped": float(decode_skipped),
                "result": result,
            })
    gaps = []
    text_changed = 0
    for line in lines:
        clock = CLOCKS_RE.search(line)
        if not clock:
            continue
        if int(clock.group(2)) == 1:
            text_changed += 1
        if clock.group(3) is not None and float(clock.group(3)) >= 0:
            gaps.append(float(clock.group(3)))
    summary: dict = {
        "vad_turn_count": sum("WS vad_turn |" in line for line in lines),
        "soft_cut_count": sum("WS soft_cut | utt=" in line and "→" in line for line in lines),
        "hard_cut_count": sum("WS hard_cut |" in line for line in lines),
        "hard_tail_confirm_count": sum("WS hard_tail_confirm |" in line for line in lines),
        "hard_tail_restore_count": sum("WS hard_tail_restore |" in line for line in lines),
        "hard_tail_discard_count": sum("WS hard_tail_discard |" in line for line in lines),
        "queue_coalesce_count": sum("reason=queue_coalesce" in line for line in lines),
        "text_changed_count": text_changed,
        "final_refresh_gap_s_max": max(gaps) if gaps else None,
    }
    for kind, jobs in metrics.items():
        completed = [job for job in jobs if job["result"] == "ready"]
        summary[f"{kind}_jobs"] = len(jobs)
        for field in (
            "audio_dur", "total_ms", "queue_ms", "infer_ms",
            "encode_ms", "prefill_ms", "decode_ms", "cache_hits", "decode_skipped",
        ):
            values = [job[field] for job in completed if job[field] >= 0]
            summary[f"{kind}_{field}_p50"] = percentile(values, 50)
            summary[f"{kind}_{field}_p95"] = percentile(values, 95)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", type=Path, default=Path(r"C:\Users\Numb\Desktop\audio.wav"))
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "ws_benchmark")
    parser.add_argument("--chunk-ms", type=int, default=20)
    parser.add_argument("--case", choices=[case[0] for case in CASES], action="append")
    args = parser.parse_args()
    if args.chunk_ms <= 0 or 1000 % args.chunk_ms:
        parser.error("--chunk-ms must be a positive divisor of 1000")
    if not args.wav.is_file():
        parser.error(f"WAV not found: {args.wav}")
    if _server_health() is not None:
        parser.error("port 8765 already has a server; stop it before the benchmark")
    pcm, sample_rate, duration = load_pcm_s16le(args.wav)
    run_dir = args.output_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    selected = [case for case in CASES if args.case is None or case[0] in args.case]
    matrix = {"wav": str(args.wav), "duration_s": round(duration, 3), "sample_rate": sample_rate,
              "chunk_ms": args.chunk_ms, "cases": {}}
    print(f"WAV {args.wav} | {duration:.2f}s | output {run_dir}", flush=True)
    for name, partial_s, long_s in selected:
        case_dir = run_dir / name
        case_dir.mkdir()
        print(f"Starting {name}: partial={partial_s}s long={long_s}s", flush=True)
        process, output = start_server(case_dir, partial_s, long_s)
        try:
            warmup = httpx.post(f"{BASE_URL}/admin/warmup", timeout=300.0)
            warmup.raise_for_status()
            summary = asyncio.run(replay(pcm, sample_rate, case_dir, args.chunk_ms))
        finally:
            stop_server(process, output)
        summary.update(summarize_log(case_dir))
        matrix["cases"][name] = summary
        (case_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (run_dir / "matrix.json").write_text(
            json.dumps(matrix, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Completed {name}: partial lag p95={summary['partial_lag_p95_s']}s; "
              f"queue p95={summary['partial_queue_ms_p95']}ms; "
              f"hard cuts={summary['hard_cut_count']}", flush=True)
    print(f"Results: {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
