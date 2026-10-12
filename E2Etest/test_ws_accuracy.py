"""CPU regression checks for endpoint assistance and final decode routing.

Extract production functions to avoid importing scipy/model runtimes. No model
accuracy is simulated; this verifies window/cache/abort and endpoint decisions.
"""
from __future__ import annotations

import ast
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from backend.base import ASRResult
from ws_overlap import realtime_window_sec


def extract(path, names, ns, class_name=None):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    body = tree.body
    if class_name:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == class_name).body
    nodes = [n for n in body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), ns)
    return ns


class EndpointTests(unittest.TestCase):
    def setUp(self):
        self.ns = extract("src/vad.py", {"chunk_rms", "advance_energy_run", "energy_onset_ready", "endpoint_energy_wait"},
                          dict(np=np, ENERGY_ONSET_RMS=0.01, ENERGY_ONSET_MS=200))

    def test_late_speech_has_time_to_confirm(self):
        run = 0
        # Voice begins 20ms before the 600ms deadline, with VAD still silent.
        for waited_ms in range(600, 800, 20):
            run = self.ns["advance_energy_run"](run, np.full(320, 0.03, dtype=np.float32))
            ready = self.ns["energy_onset_ready"](run, 16000)
            if waited_ms < 780:
                self.assertFalse(ready)
                self.assertTrue(self.ns["endpoint_energy_wait"](run, waited_ms * 16, 9600, 16000))
            else:
                self.assertTrue(ready)

    def test_noise_spike_does_not_reset_deadline(self):
        run = self.ns["advance_energy_run"](0, np.full(320, 0.03, dtype=np.float32))
        self.assertFalse(self.ns["energy_onset_ready"](run, 16000))
        run = self.ns["advance_energy_run"](run, np.zeros(320, dtype=np.float32))
        self.assertFalse(self.ns["endpoint_energy_wait"](run, 620 * 16, 9600, 16000))
        self.assertFalse(self.ns["endpoint_energy_wait"](320, 800 * 16, 9600, 16000))


class FinalTests(unittest.TestCase):
    def setUp(self):
        ns = dict(np=np, time=time, WS_FINAL_WHOLE_UTTERANCE=True,
                  REALTIME_CHUNK_SIZE_SEC=8, WS_FIXED_ENCODER_WINDOW=True,
                  WS_SHARED_UTT=True, WS_FINAL_YIELD=True, WS_FINAL_ALIGN=False,
                  WS_ASR_TEMPERATURE=0.0, TARGET_SR=16000,
                  realtime_window_sec=realtime_window_sec,
                  log=SimpleNamespace(info=lambda *a: None, warning=lambda *a: None))
        self.ns = extract("src/backend/gguf_backend.py", {"transcribe_realtime"}, ns, "QwenGGUFBackend")
        self.calls = []
        self.resumes = []
        cache = SimpleNamespace(ensure_origin=lambda *a: ["fixed-cache"],
                                get_resume=lambda *a: {"next_idx": 1},
                                set_resume=lambda *a: self.resumes.append(a[-1]))
        def run(*args, **kwargs):
            self.calls.append((args, kwargs))
            return ASRResult("正文。", "zh", perf={"n_chunks": 1})
        self.backend = SimpleNamespace(_utt_cache=cache, _run=run,
                                       advance_encoder=lambda *a: None)

    def invoke(self, final=True, seconds=14, abort=None, yield_event=None):
        return self.ns["transcribe_realtime"](
            self.backend, np.zeros(int(seconds * 16000), dtype=np.float32), 16000,
            is_final=final, abort_event=abort, yield_event=yield_event,
        )

    def test_whole_final_does_not_mix_fixed_cache_or_resume(self):
        self.invoke(yield_event=threading.Event())
        args, kw = self.calls[0]
        self.assertEqual(args[3], 14.0)
        self.assertIsNone(kw["encoder_cache"])
        self.assertEqual(kw["skip_decode_before"], 0)
        self.assertIsNone(kw["yield_event"])
        self.assertIsNone(self.resumes[-1])

    def test_partial_and_short_final_keep_existing_window(self):
        for final, seconds in ((False, 14), (True, 4)):
            self.calls.clear()
            self.invoke(final, seconds)
            self.assertEqual(self.calls[0][0][3], 8.0)
            self.assertEqual(self.calls[0][1]["encoder_cache"], ["fixed-cache"])

    def test_empty_whole_result_falls_back(self):
        def run(*args, **kw):
            self.calls.append((args, kw))
            return ASRResult("" if len(self.calls) == 1 else "回退。", "zh", perf={})
        self.backend._run = run
        self.assertEqual(self.invoke().text, "回退。")
        self.assertEqual([a[3] for a, _ in self.calls], [14.0, 8.0])

    def test_exception_falls_back_but_abort_does_not(self):
        abort = threading.Event()
        def run(*args, **kw):
            self.calls.append((args, kw))
            if len(self.calls) == 1:
                raise RuntimeError("encoder failure")
            return ASRResult("回退。", "zh", perf={})
        self.backend._run = run
        self.assertEqual(self.invoke(abort=abort).text, "回退。")
        self.calls.clear()
        abort.set()
        with self.assertRaises(RuntimeError):
            self.invoke(abort=abort)
        self.assertEqual(len(self.calls), 1)

    def test_switch_off_restores_fixed_final(self):
        self.ns["WS_FINAL_WHOLE_UTTERANCE"] = False
        self.invoke()
        self.assertEqual(self.calls[0][0][3], 8.0)


if __name__ == "__main__":
    unittest.main()
