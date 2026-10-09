"""System-tray controller for the green-pack Qwen ASR service."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

if not getattr(sys, "frozen", False):
    _SRC = Path(__file__).resolve().parent
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

from singleton import TRAY_MUTEX, try_acquire  # noqa: E402
import pystray  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

CREATE_NO_WINDOW = 0x08000000
_POLL_S = 2.0
_START_WAIT_S = 30.0
_STOP_WAIT_S = 15.0
_WARMUP_S = 300.0
_RESTART_COOLDOWN_S = 3.0
_RESTART_MAX = 5
_LNK_NAME = "ASRTray.lnk"


def _pack_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    here = Path(__file__).resolve().parent
    # src/tray_app.py in repo, or app/src/tray_app.py in a pack
    code = here.parent
    pack = code.parent
    if (pack / "runtime" / "python.exe").is_file() and (code / "run.py").is_file():
        return pack
    return code


def _load_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


class TrayService:
    def __init__(self) -> None:
        self.root = _pack_root()
        env = _load_env_file(self.root / ".env")
        self.host = env.get("HOST", "127.0.0.1")
        self.port = int(env.get("PORT", "8765") or "8765")
        self.base = f"http://{self.host}:{self.port}"
        self.proc: subprocess.Popen | None = None
        self._log_fp = None
        self._lock = threading.Lock()
        self._crash_streak = 0
        self._auto_restart = True
        self._last_state = "down"
        self._last_error = ""
        self._icon = None
        self._stop_poll = threading.Event()
        self._kill_service_on_exit = False

    def service_cmd(self) -> list[str] | None:
        runtime = self.root / "runtime" / "python.exe"
        script = self.root / "app" / "run.py"
        if runtime.is_file() and script.is_file():
            return [str(runtime), str(script)]
        repo_run = self.root / "run.py"
        if repo_run.is_file():
            return [sys.executable, str(repo_run)]
        return None

    def logs_dir(self) -> Path:
        env = _load_env_file(self.root / ".env")
        raw = env.get("LOG_DIR", "")
        if raw:
            p = Path(raw).expanduser()
            return p if p.is_absolute() else (self.root / p)
        if (self.root / "runtime" / "python.exe").is_file():
            return self.root / "logs"
        return self.root / "vendor" / "logs"

    def _http_json(self, method: str, path: str, timeout: float):
        req = urllib.request.Request(
            self.base + path,
            method=method,
            headers={"Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                return json.loads(body) if body else {}
        except Exception as exc:
            self._last_error = str(exc)
            return None

    def health(self):
        return self._http_json("GET", "/health", timeout=2.0)

    def warmup_async(self) -> None:
        def _run():
            self._http_json("POST", "/admin/warmup", timeout=_WARMUP_S)

        threading.Thread(target=_run, daemon=True).start()

    def is_running(self) -> bool:
        with self._lock:
            return self.proc is not None and self.proc.poll() is None

    def start(self) -> str:
        with self._lock:
            if self.proc is not None and self.proc.poll() is None:
                return "already"
            if self.health() is not None:
                return "already"
            cmd = self.service_cmd()
            if not cmd:
                self._last_error = "runtime/python.exe or app/run.py not found"
                return "error"
            log_dir = self.logs_dir()
            log_dir.mkdir(parents=True, exist_ok=True)
            if self._log_fp:
                try:
                    self._log_fp.close()
                except Exception:
                    pass
            self._log_fp = open(log_dir / "service.log", "a", encoding="utf-8")
            flags = 0
            startupinfo = None
            if sys.platform == "win32":
                flags = CREATE_NO_WINDOW
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = 0
            try:
                self.proc = subprocess.Popen(
                    cmd,
                    cwd=str(self.root),
                    stdout=self._log_fp,
                    stderr=subprocess.STDOUT,
                    startupinfo=startupinfo,
                    creationflags=flags,
                )
            except Exception as exc:
                self._last_error = str(exc)
                return "error"
        deadline = time.time() + _START_WAIT_S
        while time.time() < deadline:
            if self.health() is not None:
                self._crash_streak = 0
                self.warmup_async()
                return "ok"
            with self._lock:
                if self.proc is not None and self.proc.poll() is not None:
                    self._last_error = f"service exited with code {self.proc.returncode}"
                    return "error"
            time.sleep(0.4)
        self._last_error = "timeout waiting for /health"
        return "error"

    def stop(self) -> None:
        self._http_json("POST", "/admin/shutdown", timeout=5.0)
        deadline = time.time() + _STOP_WAIT_S
        while time.time() < deadline:
            with self._lock:
                if self.proc is None or self.proc.poll() is not None:
                    break
            if self.health() is None:
                break
            time.sleep(0.3)
        with self._lock:
            if self.proc is not None and self.proc.poll() is None:
                pid = self.proc.pid
                if sys.platform == "win32":
                    subprocess.run(
                        ["taskkill", "/PID", str(pid), "/T", "/F"],
                        capture_output=True,
                        creationflags=CREATE_NO_WINDOW,
                    )
                else:
                    self.proc.terminate()
                    try:
                        self.proc.wait(timeout=5)
                    except Exception:
                        self.proc.kill()
            self.proc = None

    def restart(self) -> str:
        self.stop()
        time.sleep(0.5)
        return self.start()

    def tray_exe(self) -> Path:
        if getattr(sys, "frozen", False):
            return Path(sys.executable).resolve()
        return Path(__file__).resolve()

    def startup_lnk(self) -> Path:
        appdata = Path(os.environ.get("APPDATA", ""))
        return appdata / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup" / _LNK_NAME

    def autostart_enabled(self) -> bool:
        return self.startup_lnk().is_file()

    def _write_lnk(self) -> None:
        try:
            import win32com.client  # type: ignore
        except ImportError:
            self._last_error = "pywin32 is required for autostart"
            return
        lnk = self.startup_lnk()
        lnk.parent.mkdir(parents=True, exist_ok=True)
        shell = win32com.client.Dispatch("WScript.Shell")
        sc = shell.CreateShortCut(str(lnk))
        sc.Targetpath = str(self.tray_exe())
        sc.WorkingDirectory = str(self.root)
        sc.WindowStyle = 7
        sc.Description = "ASR tray"
        sc.save()

    def _lnk_target(self) -> str:
        lnk = self.startup_lnk()
        if not lnk.is_file():
            return ""
        try:
            import win32com.client  # type: ignore

            shell = win32com.client.Dispatch("WScript.Shell")
            sc = shell.CreateShortCut(str(lnk))
            return str(sc.Targetpath or "")
        except Exception:
            return ""

    def repair_autostart_if_needed(self) -> None:
        if not self.autostart_enabled():
            return
        want = str(self.tray_exe())
        got = self._lnk_target()
        if os.path.normcase(got) != os.path.normcase(want):
            self._write_lnk()

    def set_autostart(self, enabled: bool) -> None:
        lnk = self.startup_lnk()
        if enabled:
            self._write_lnk()
        elif lnk.exists():
            lnk.unlink()

    def status_label(self) -> str:
        data = self.health()
        if data is None:
            if self._last_error:
                return f"异常（{self._last_error[:40]}）"
            return "未启动"
        state = str(data.get("state") or "")
        if state == "READY":
            return "就绪"
        if state == "LOADING":
            return "模型加载中"
        if state == "UNLOADED":
            return "空闲（模型未加载）"
        if state == "UNLOADING":
            return "正在卸载模型"
        return state or "运行中"

    def icon_color(self) -> tuple[int, int, int]:
        data = self.health()
        if data is None:
            if self.is_running():
                return (220, 50, 50)
            return (160, 160, 160)
        state = str(data.get("state") or "")
        if state == "READY":
            return (40, 180, 70)
        if state == "LOADING":
            return (230, 180, 40)
        if state == "UNLOADED":
            return (230, 230, 230)
        if state == "UNLOADING":
            return (230, 180, 40)
        return (160, 160, 160)

    def make_icon(self):
        color = self.icon_color()
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.ellipse((8, 8, 56, 56), fill=color + (255,))
        return img

    def tooltip(self) -> str:
        return f"ASR\n{self.status_label()}\n端口: {self.port}"

    def _watch_crashes(self) -> None:
        with self._lock:
            proc = self.proc
        if proc is None:
            return
        code = proc.poll()
        if code is None:
            return
        if not self._auto_restart:
            return
        self._crash_streak += 1
        if self._crash_streak > _RESTART_MAX:
            self._last_error = f"service crashed {self._crash_streak} times"
            self._auto_restart = False
            return
        time.sleep(_RESTART_COOLDOWN_S)
        self.start()

    def poll_loop(self) -> None:
        while not self._stop_poll.is_set():
            try:
                self._watch_crashes()
                if self._icon is not None:
                    self._icon.icon = self.make_icon()
                    self._icon.title = self.tooltip()
            except Exception:
                pass
            self._stop_poll.wait(_POLL_S)

    def _notify(self, icon) -> None:
        self._icon = icon
        icon.icon = self.make_icon()
        icon.title = self.tooltip()

    def open_logs(self, _icon=None, _item=None) -> None:
        path = self.logs_dir()
        path.mkdir(parents=True, exist_ok=True)
        os.startfile(str(path))  # noqa: S606

    def open_root(self, _icon=None, _item=None) -> None:
        os.startfile(str(self.root))  # noqa: S606

    def on_start(self, _icon=None, _item=None) -> None:
        self._auto_restart = True
        threading.Thread(target=self.start, daemon=True).start()

    def on_stop(self, _icon=None, _item=None) -> None:
        self._auto_restart = False
        threading.Thread(target=self.stop, daemon=True).start()

    def on_restart(self, _icon=None, _item=None) -> None:
        self._auto_restart = True
        threading.Thread(target=self.restart, daemon=True).start()

    def on_toggle_autostart(self, _icon=None, _item=None) -> None:
        self.set_autostart(not self.autostart_enabled())

    def _quit_icon(self, icon) -> None:
        self._auto_restart = False
        self._stop_poll.set()
        try:
            icon.visible = False
        except Exception:
            pass
        try:
            icon.stop()
        except Exception:
            pass
        if sys.platform == "win32":
            try:
                import ctypes

                ctypes.windll.user32.PostQuitMessage(0)
            except Exception:
                pass

    def on_exit_tray(self, icon=None, _item=None) -> None:
        """Close tray only; ASR python process keeps running."""
        self._kill_service_on_exit = False
        if icon is not None:
            self._quit_icon(icon)

    def on_exit_all(self, icon=None, _item=None) -> None:
        """Close tray, then stop the ASR python process after the UI loop ends."""
        self._kill_service_on_exit = True
        if icon is not None:
            self._quit_icon(icon)

    def build_menu(self):
        return pystray.Menu(
            pystray.MenuItem(lambda _: f"ASR  ·  {self.status_label()}", None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("启动服务", self.on_start),
            pystray.MenuItem("停止服务", self.on_stop),
            pystray.MenuItem("重启服务", self.on_restart),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("打开日志目录", self.open_logs),
            pystray.MenuItem("打开程序目录", self.open_root),
            pystray.MenuItem(
                "开机自动启动",
                self.on_toggle_autostart,
                checked=lambda _: self.autostart_enabled(),
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("退出托盘（服务继续运行）", self.on_exit_tray),
            pystray.MenuItem("退出并停止服务", self.on_exit_all, default=True),
        )

    def run(self) -> int:
        if not try_acquire(TRAY_MUTEX):
            return 0
        self.repair_autostart_if_needed()
        self.start()
        threading.Thread(target=self.poll_loop, daemon=True).start()
        icon = pystray.Icon(
            "ASR",
            self.make_icon(),
            self.tooltip(),
            menu=self.build_menu(),
        )
        self._icon = icon
        icon.run()
        if self._kill_service_on_exit:
            self.stop()
        os._exit(0)


def main() -> int:
    return TrayService().run()


if __name__ == "__main__":
    raise SystemExit(main())
