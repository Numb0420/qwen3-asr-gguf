"""Submit a local wav as an offline HTTP task.

    conda activate lingting
    python E2Etest/test_api_http.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx

# --- 在这里改录音文件 ---
WAV_PATH = r"C:\Users\Numb\Desktop\11s.wav"
LANGUAGE = "auto"  # zh / en / auto
BASE_URL = "http://127.0.0.1:8765"
POLL_S = 1.0
TIMEOUT_S = 300.0


def main() -> int:
    wav = Path(WAV_PATH).expanduser()
    if not wav.is_file():
        print(f"wav not found: {wav}", file=sys.stderr)
        return 1

    timeout = httpx.Timeout(TIMEOUT_S, connect=5.0)
    with httpx.Client(base_url=BASE_URL, timeout=timeout) as client:
        try:
            health = client.get("/health")
            health.raise_for_status()
        except Exception as exc:
            print(f"server unreachable at {BASE_URL}: {exc}", file=sys.stderr)
            return 1
        print("health:", health.json())

        print(f"POST /offline/transcribe-path | file={wav} language={LANGUAGE}")
        resp = client.post(
            "/offline/transcribe-path",
            json={"audio_path": str(wav), "language": LANGUAGE},
        )
        print("submit status:", resp.status_code, resp.text)
        if resp.status_code != 200:
            return 1
        ack = resp.json()
        task_id = ack["task_id"]
        print("task_id:", task_id)

        early = client.get(f"/offline/tasks/{task_id}/result")
        print("early result status:", early.status_code)
        if early.status_code not in (200, 409):
            print("unexpected /result while polling:", early.status_code, early.text, file=sys.stderr)
            return 1

        deadline = time.time() + TIMEOUT_S
        while time.time() < deadline:
            st = client.get(f"/offline/tasks/{task_id}")
            st.raise_for_status()
            body = st.json()
            print("status:", body.get("status"), body.get("message"), body.get("files"))
            if body.get("status") == "completed":
                result = client.get(f"/offline/tasks/{task_id}/result")
                print("result status:", result.status_code)
                data = result.json()
                print("response:", data)
                for seg in data.get("segments") or []:
                    print(f"  [{seg.get('index')}] {seg.get('start')}-{seg.get('end')} {seg.get('text')}")
                    for ch in seg.get("chars") or []:
                        print(f"    {ch.get('start'):.3f}-{ch.get('end'):.3f} {ch.get('text')}")
                return 0
            if body.get("status") == "failed":
                print("task failed:", body.get("message"), file=sys.stderr)
                return 1
            time.sleep(POLL_S)
        print("poll timeout", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
