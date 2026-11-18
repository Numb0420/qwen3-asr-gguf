"""Launcher checks without starting the actual ASR service or opening a tray."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "build/tray-deps"))
from tray_app import TrayService


class LauncherTests(unittest.TestCase):
    def test_initialization_never_loads_https_certificates(self):
        with patch("ssl.create_default_context", side_effect=FileNotFoundError(2, "missing CA file")):
            service = TrayService()
        self.assertFalse(any(hasattr(handler, "https_open") for handler in service._http.handlers))

    def test_wildcard_host_uses_loopback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text("HOST=0.0.0.0\nPORT=8765", encoding="utf-8")
            with patch("tray_app._pack_root", return_value=root):
                self.assertEqual(TrayService().base, "http://127.0.0.1:8765")

    def test_configured_python_is_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "run.py").touch()
            python = root / "python.exe"
            python.touch()
            (root / ".env").write_text(f"ASR_PYTHON={python}", encoding="utf-8")
            with patch("tray_app._pack_root", return_value=root), patch.dict("os.environ", {"ASR_PYTHON": str(python)}):
                service = TrayService()
                with patch("tray_app.subprocess.run", return_value=Mock(returncode=0)):
                    self.assertEqual(service.service_cmd(), [str(python), str(root / "run.py")])

    def test_exit_requests_service_shutdown(self):
        service = TrayService()
        icon = Mock()
        service.on_exit_all(icon)
        self.assertTrue(service._kill_service_on_exit)
        self.assertTrue(service._stop_poll.is_set())
        self.assertFalse(service._auto_restart)
        icon.stop.assert_called_once()

    def test_shutdown_blocks_late_start(self):
        service = TrayService()
        service._stop_poll.set()
        with patch.object(service, "service_cmd") as command:
            self.assertEqual(service.start(), "stopped")
            command.assert_not_called()

    def test_health_failure_preserves_startup_error(self):
        service = TrayService()
        service._last_error = "missing service Python"
        with patch.object(service._http, "open", side_effect=FileNotFoundError(2, "no such file")):
            self.assertIsNone(service.health())
        self.assertEqual(service._last_error, "missing service Python")

    def test_user_conda_environment_is_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "run.py").touch()
            python = root / ".conda/envs/lingting/python.exe"
            python.parent.mkdir(parents=True)
            python.touch()
            with patch("tray_app._pack_root", return_value=root), patch("tray_app.Path.home", return_value=root):
                service = TrayService()
                with patch("tray_app.subprocess.run", return_value=Mock(returncode=0)):
                    self.assertEqual(service.service_cmd()[0], str(python))


if __name__ == "__main__":
    unittest.main()
