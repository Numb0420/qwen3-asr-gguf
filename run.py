"""Start the local GGUF ASR server.

Dev (conda lingting):

    python run.py

Green pack:

    runtime\\python.exe app\\run.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_RUN_PY = Path(__file__).resolve()
_CODE_ROOT = _RUN_PY.parent  # repo root, or app/ in a green pack
_PACK_ROOT = _CODE_ROOT.parent
_IS_PACK = _CODE_ROOT.name == "app" and (_PACK_ROOT / "runtime" / "python.exe").is_file()
ROOT = _PACK_ROOT if _IS_PACK else _CODE_ROOT

_LINGTING = Path(r"D:\anaconda3\envs\lingting\python.exe")
if (
    not _IS_PACK
    and not getattr(sys, "frozen", False)
    and _LINGTING.exists()
    and Path(sys.executable).resolve() != _LINGTING.resolve()
):
    os.execv(str(_LINGTING), [str(_LINGTING), str(_RUN_PY), *sys.argv[1:]])

os.chdir(ROOT)
sys.path.insert(0, str(_CODE_ROOT / "src"))
sys.path.insert(0, str(_CODE_ROOT / "vendor"))


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env(ROOT / ".env")

from singleton import SERVER_MUTEX, try_acquire  # noqa: E402

if __name__ == "__main__":
    if not try_acquire(SERVER_MUTEX):
        print("Qwen ASR service is already running")
        raise SystemExit(0)

from config import HOST, PORT, apply_llm_device_env  # noqa: E402

apply_llm_device_env()
import uvicorn  # noqa: E402
import server as server_mod  # noqa: E402


if __name__ == "__main__":
    config = uvicorn.Config("server:app", host=HOST, port=PORT, reload=False)
    uv = uvicorn.Server(config)
    server_mod.bind_uvicorn_server(uv)
    uv.run()
