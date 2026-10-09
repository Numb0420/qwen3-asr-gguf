from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
import heapq
import threading
import time
from typing import Any, Callable, Literal

JobKind = Literal["normal", "partial", "final"]


class InferAborted(Exception):
    """Job was cancelled via abort_session (offline cancel / drop pending)."""


_infer_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="qwen3-asr-infer",
)


@dataclasses.dataclass(order=True)
class _InferJob:
    priority: int
    submit_time: float
    seq: int
    future: asyncio.Future = dataclasses.field(compare=False)
    fn: Callable = dataclasses.field(compare=False)
    kind: JobKind = dataclasses.field(default="normal", compare=False)
    # 身份：用于 final 抢占同 utt 的 partial。离线任务 session_id="" / utt_id=0。
    session_id: str = dataclasses.field(default="", compare=False)
    utt_id: int = dataclasses.field(default=0, compare=False)
    # 每个 job 独立 abort_event；final 入队时 set 同 utt 正在跑的 partial 的 event，
    # partial 的 decode 循环命中后尽快退出。job 生命周期结束自然失效，无需手工 reset。
    abort_event: threading.Event = dataclasses.field(
        default_factory=threading.Event, compare=False
    )
    yield_event: threading.Event = dataclasses.field(
        default_factory=threading.Event, compare=False
    )


class PriorityInferQueue:
    """Single-worker queue. WS (0) beats HTTP (1).

    final 抢占：final 入队前 drop pending 同 utt partial + set running 同 utt partial 的
    abort_event，让正在 decode 的 partial 尽快退出，腾出 worker 给 final。
    离线任务（kind=normal）和其他会话不受影响。
    """

    def __init__(self):
        self._heap: list[_InferJob] = []
        self._lock = asyncio.Lock()
        self._has_work = asyncio.Event()
        self._worker_task: asyncio.Task | None = None
        self._running = 0
        self._seq = 0
        self._idle = asyncio.Event()
        self._idle.set()
        # _running_job 由 worker（event loop 线程）写、submit（event loop 线程）读，
        # 用 threading.Lock 防御未来 worker 逻辑移进线程池线程的潜在竞态。
        self._running_lock = threading.Lock()
        self._running_job: _InferJob | None = None
        self._idle_callbacks: list = []

    def start(self):
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker())

    def stop(self):
        if self._worker_task:
            self._worker_task.cancel()
            self._worker_task = None

    @property
    def busy(self) -> bool:
        return self._running > 0 or bool(self._heap)

    async def wait_idle(self) -> None:
        await self._idle.wait()

    async def run_blocking(self, fn: Callable, priority: int = 0) -> Any:
        """Run a blocking fn on the inference thread (load/unload)."""
        return await self.submit(fn, priority=priority, kind="final")

    async def _worker(self):
        loop = asyncio.get_event_loop()
        while True:
            await self._has_work.wait()
            async with self._lock:
                if not self._heap:
                    self._has_work.clear()
                    if self._running == 0:
                        self._idle.set()
                    continue
                job = heapq.heappop(self._heap)
                self._running += 1
                self._idle.clear()
            with self._running_lock:
                self._running_job = job
            try:
                result = await loop.run_in_executor(_infer_executor, job.fn)
                if not job.future.done():
                    job.future.set_result(result)
            except Exception as e:
                if not job.future.done():
                    job.future.set_exception(e)
            finally:
                with self._running_lock:
                    if self._running_job is job:
                        self._running_job = None
                async with self._lock:
                    self._running -= 1
                    if self._running == 0 and not self._heap:
                        self._idle.set()
                        self._has_work.clear()
                        callbacks = list(self._idle_callbacks)
                    else:
                        callbacks = []
                for cb in callbacks:
                    try:
                        loop.call_soon(cb)
                    except Exception:
                        pass

    async def submit(
        self,
        fn: Callable,
        priority: int = 1,
        kind: JobKind = "normal",
        session_id: str = "",
        utt_id: int = 0,
        abort_event: threading.Event | None = None,
        yield_event: threading.Event | None = None,
    ) -> Any:
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        async with self._lock:
            if kind in ("partial", "final") and session_id:
                # 1. drop pending 同 utt partial（合并旧 partial）
                self._drop_pending_partials(session_id, utt_id)
                # 2. abort running 同 utt partial（final 抢占）
                if kind == "final":
                    with self._running_lock:
                        running = self._running_job
                        if (
                            running is not None
                            and running.kind == "partial"
                            and running.session_id == session_id
                            and running.utt_id == utt_id
                        ):
                            running.abort_event.set()
                elif kind == "partial":
                    with self._running_lock:
                        running = self._running_job
                        if (
                            running is not None
                            and running.kind == "final"
                            and running.session_id == session_id
                            and running.utt_id != utt_id
                            and running.yield_event is not None
                        ):
                            running.yield_event.set()
            self._seq += 1
            job = _InferJob(
                priority=priority,
                submit_time=time.time(),
                seq=self._seq,
                future=future,
                fn=fn,
                kind=kind,
                session_id=session_id,
                utt_id=utt_id,
                abort_event=abort_event if abort_event is not None else threading.Event(),
                yield_event=yield_event if yield_event is not None else threading.Event(),
            )
            heapq.heappush(self._heap, job)
            self._idle.clear()
            self._has_work.set()
        return await future

    def _drop_pending_partials(self, session_id: str, utt_id: int) -> None:
        kept: list[_InferJob] = []
        for job in self._heap:
            if (
                job.kind == "partial"
                and job.session_id == session_id
                and job.utt_id == utt_id
                and not job.future.done()
            ):
                job.future.set_result(None)
                continue
            kept.append(job)
        self._heap = kept
        heapq.heapify(self._heap)

    def add_idle_callback(self, cb) -> None:
        self._idle_callbacks.append(cb)

    def remove_idle_callback(self, cb) -> None:
        self._idle_callbacks = [item for item in self._idle_callbacks if item is not cb]

    async def abort_session(self, session_id: str) -> int:
        """Abort running job + drop pending jobs for session_id. Returns affected count."""
        if not session_id:
            return 0
        n = 0
        async with self._lock:
            with self._running_lock:
                running = self._running_job
                if (
                    running is not None
                    and running.session_id == session_id
                    and not running.abort_event.is_set()
                ):
                    running.abort_event.set()
                    n += 1
            kept: list[_InferJob] = []
            for job in self._heap:
                if job.session_id == session_id and not job.future.done():
                    job.abort_event.set()
                    job.future.set_exception(InferAborted(f"session {session_id} aborted"))
                    n += 1
                    continue
                kept.append(job)
            self._heap = kept
            heapq.heapify(self._heap)
            if self._running == 0 and not self._heap:
                self._idle.set()
                self._has_work.clear()
        return n


infer_queue = PriorityInferQueue()
