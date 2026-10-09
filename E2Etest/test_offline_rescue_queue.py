"""CPU regression checks using actual task methods, without loading ASR dependencies.

The task methods are extracted from source because importing offline_tasks also
imports scipy, VAD and hotword runtimes. These tests exercise orchestration only.
"""
from __future__ import annotations

import ast
import asyncio
from dataclasses import replace
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from backend.base import WordTimestamp
from inference_queue import PriorityInferQueue
from offline_split import SplitConfig, plan_chunks
from offline_stitch import append_offline_chunk, finalize_offline_chunk
from segments import has_transcript_content, words_to_segments


def task_methods():
    tree = ast.parse((ROOT / "src/offline_tasks.py").read_text(encoding="utf-8"))
    wanted = {"_dedup_rescue_overlap", "_slice_audio"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OfflineTaskStore")
    methods = {"_raise_if_cancelled", "_call_infer", "_wait_infer", "_infer_split"}
    body = [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in methods]
    nodes.append(ast.ClassDef(name="Store", bases=[], keywords=[], body=body, decorator_list=[]))
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    ns = dict(np=np, asyncio=asyncio, WordTimestamp=WordTimestamp,
              _RESCUE_DEDUP_TOL=0.1, FILE_CHUNK_SIZE_SEC=40,
              REQUEST_TIMEOUT=0.03, replace=replace, plan_chunks=plan_chunks,
              _offline_split_config=SplitConfig, OfflineCancelled=RuntimeError,
              finalize_offline_chunk=finalize_offline_chunk,
              has_transcript_content=has_transcript_content,
              append_offline_chunk=append_offline_chunk)
    exec(compile(module, str(ROOT / "src/offline_tasks.py"), "exec"), ns)
    return ns


class RescueTests(unittest.IsolatedAsyncioTestCase):
    def test_punctuation_only_rescue_is_empty(self):
        dedup = task_methods()["_dedup_rescue_overlap"]
        words = [WordTimestamp("正文", 29.0, 29.2), WordTimestamp("。", 29.6, 29.6)]
        self.assertEqual(dedup("正文。", "正文。", words, 29.4), ("", []))
        self.assertEqual(dedup("正文。", " 。 ！ ", None, 29.4), ("", []))
        text, kept = dedup("正文。", "对。", [WordTimestamp("对。", 30.0, 30.2)], 29.4)
        self.assertEqual(text, "对。")
        self.assertEqual(len(kept), 1)

    def test_duplicate_punctuation_cannot_create_zero_time_subtitle(self):
        words = [WordTimestamp("还是正常", 344.448, 344.848),
                 WordTimestamp("。", 344.848, 344.848),
                 WordTimestamp("。", 345.0, 345.0),
                 WordTimestamp("对", 346.0, 346.2), WordTimestamp("。", 346.2, 346.2)]
        segments = words_to_segments(words)
        self.assertEqual([s["text"] for s in segments], ["还是正常。", "对。"])
        self.assertEqual([s["index"] for s in segments], [1, 2])
        self.assertEqual([s["start"] for s in segments], [344.448, 346.0])
        self.assertEqual(words_to_segments([WordTimestamp("。", 345.0, 345.0)]), [])
        self.assertEqual(words_to_segments(None, " 。 ！ "), [])
        self.assertEqual(words_to_segments(None, "对。")[0]["text"], "对。")

    def test_rescue_seam_uses_chunk_coordinates(self):
        ns = task_methods()
        cutoff = 344.768 - 315.408
        words = [WordTimestamp("old", cutoff - 0.7, cutoff - 0.2),
                 WordTimestamp("new", cutoff + 0.2, cutoff + 0.6)]
        text, kept = ns["_dedup_rescue_overlap"]("old", "oldnew", words, cutoff)
        self.assertEqual(text, "new")
        self.assertEqual(kept, words[1:])
        # Check both real call sites also pass the relative cutoff.
        tree = ast.parse((ROOT / "src/offline_tasks.py").read_text(encoding="utf-8"))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "_dedup_rescue_overlap"]
        self.assertEqual(len(calls), 2)
        for call in calls:
            self.assertEqual(ast.unparse(call.args[-1]), "last_char_end - plan.audio_start")

    async def test_split_rescue_silence_and_hard_overlap(self):
        for silences, expected_end, expected_start in (
            ([(116.0, 116.6)], 16.3, 15.9), ([], 20.0, 18.5),
        ):
            ns = task_methods()
            store = ns["Store"]()
            calls = []
            async def infer(task, audio, sr, language, chunk_size):
                calls.append((float(audio[0]) / sr, len(audio) / sr))
                text = chr(65 + len(calls) - 1)
                return SimpleNamespace(text=text, words=[WordTimestamp(text, 2.0, 2.5)])
            store._call_infer = infer
            task = SimpleNamespace(task_id="split", status="running", cancel_event=threading.Event())
            text, words = await store._infer_split(
                task, np.arange(4000, dtype=np.float32), 100, None,
                silences=silences, audio_start=100.0,
            )
            self.assertAlmostEqual(calls[0][1], expected_end)
            self.assertAlmostEqual(calls[1][0], expected_start)
            self.assertEqual(len(calls), len(text))
            self.assertEqual(text, "".join(w.word for w in words))
            self.assertAlmostEqual(words[1].start, expected_start + 2.0)

    async def test_timeout_aborts_session(self):
        ns = task_methods()
        aborted = []
        async def abort(session):
            aborted.append(session)
        ns["infer_queue"] = SimpleNamespace(abort_session=abort)
        store = ns["Store"]()
        async def infer(*args, **kwargs):
            await asyncio.Event().wait()
        store._infer = infer
        task = SimpleNamespace(task_id="timeout", status="running", cancel_event=threading.Event())
        with self.assertRaises(asyncio.TimeoutError):
            await store._call_infer(task, None, 16000, None, 40)
        self.assertTrue(task.cancel_event.is_set())
        self.assertEqual(aborted, ["timeout"])

    async def test_running_inference_receives_timeout_abort(self):
        ns = task_methods()
        queue = PriorityInferQueue()
        ns["infer_queue"] = queue
        store = ns["Store"]()
        task = SimpleNamespace(task_id="running", status="running", cancel_event=threading.Event())
        started = threading.Event()
        stopped = threading.Event()
        def job():
            started.set()
            task.cancel_event.wait(1)
            stopped.set()
        async def infer(*args, **kwargs):
            return await queue.submit(job, session_id=task.task_id, abort_event=task.cancel_event)
        store._infer = infer
        queue.start()
        try:
            with self.assertRaises(asyncio.TimeoutError):
                await store._call_infer(task, None, 16000, None, 40)
            await asyncio.wait_for(queue.wait_idle(), 1)
            self.assertTrue(started.is_set())
            self.assertTrue(stopped.is_set())
            self.assertFalse(queue.busy)
            # The same shared queue must still accept subsequent work.
            self.assertEqual(await queue.submit(lambda: "next"), "next")
        finally:
            task.cancel_event.set()
            worker = queue._worker_task
            queue.stop()
            if worker:
                await asyncio.gather(worker, return_exceptions=True)


class QueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_pending_job_is_not_executed(self):
        queue = PriorityInferQueue()
        ran = []
        pending = asyncio.create_task(queue.submit(lambda: ran.append("bad")))
        await asyncio.sleep(0)
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        queue.start()
        try:
            await asyncio.wait_for(queue.wait_idle(), 1)
            self.assertEqual(ran, [])
        finally:
            worker = queue._worker_task
            queue.stop()
            if worker:
                await asyncio.gather(worker, return_exceptions=True)

    async def test_abort_removes_cancelled_future(self):
        queue = PriorityInferQueue()
        pending = asyncio.create_task(queue.submit(lambda: None, session_id="offline"))
        await asyncio.sleep(0)
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        self.assertEqual(await queue.abort_session("offline"), 1)
        self.assertEqual(queue._heap, [])
        self.assertFalse(queue.busy)


if __name__ == "__main__":
    unittest.main()
