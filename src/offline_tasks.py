"""In-memory offline transcription tasks."""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Awaitable

import numpy as np
from scipy.signal import resample_poly

from config import (
    ASR_REMOVE_FILLERS,
    ENABLE_ALIGNER,
    FILE_CHUNK_SIZE_SEC,
    OFFLINE_CUT_EPS_SECONDS,
    OFFLINE_FALLBACK_SILENCE_MS,
    OFFLINE_HARD_CUT_OVERLAP_SECONDS,
    OFFLINE_SILENCE_OVERLAP_SECONDS,
    OFFLINE_MIN_SILENCE_MS,
    OFFLINE_SAVE_RESULT,
    OFFLINE_SPLIT_SEARCH_SECONDS,
    OFFLINE_TARGET_CHUNK_SECONDS,
    OUTPUT_DIR,
    REQUEST_TIMEOUT,
    TARGET_SR,
)
from backend.base import WordTimestamp
from fillers import clean_filler_segments
from logger import log
from offline_split import SplitConfig, plan_chunks, stitch_chunk_text
from offline_stitch import append_offline_chunk, finalize_offline_chunk
from hotword_correct import apply_hotword
from itn import apply_chinese_itn
from segments import words_to_segments
from vad import scan_silences

InferFn = Callable[..., Awaitable]


def _new_task_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def _decode_audio_file(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sr = sf.read(str(path))
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if int(sr) != TARGET_SR:
        from math import gcd

        g = gcd(int(sr), TARGET_SR)
        audio = resample_poly(audio, TARGET_SR // g, int(sr) // g).astype(np.float32)
        sr = TARGET_SR
    return audio, int(sr)


def _offline_split_config() -> SplitConfig:
    return SplitConfig(
        target=OFFLINE_TARGET_CHUNK_SECONDS,
        search=OFFLINE_SPLIT_SEARCH_SECONDS,
        max_chunk=FILE_CHUNK_SIZE_SEC,
        min_silence=OFFLINE_MIN_SILENCE_MS / 1000.0,
        fallback_silence=OFFLINE_FALLBACK_SILENCE_MS / 1000.0,
        hard_overlap=OFFLINE_HARD_CUT_OVERLAP_SECONDS,
        silence_overlap=OFFLINE_SILENCE_OVERLAP_SECONDS,
        cut_eps=OFFLINE_CUT_EPS_SECONDS,
        allow_hard_overlap=ENABLE_ALIGNER,
    )


def _slice_audio(audio: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    i0 = int(round(start * sr))
    i1 = int(round(end * sr))
    i0 = max(0, min(i0, int(audio.size)))
    i1 = max(i0, min(i1, int(audio.size)))
    return audio[i0:i1]


def _speech_stats(
    silences: list[tuple[float, float]],
    t_start: float,
    t_end: float,
) -> dict:
    """Return speech coverage stats for [t_start, t_end] using silence regions.

    silences are non-overlapping, sorted (start, end) pairs from scan_silences.
    Returns speech total plus max_speech (longest continuous speech segment).
    """
    duration = max(0.0, t_end - t_start)
    if duration <= 0:
        return {
            "duration": 0.0, "silence": 0.0, "speech": 0.0,
            "speech_ratio": 0.0, "max_speech": 0.0,
        }
    covered = 0.0
    max_speech = 0.0
    prev_end = t_start
    for s, e in silences:
        lo, hi = max(s, t_start), min(e, t_end)
        if hi > lo:
            covered += hi - lo
            if lo > prev_end:
                max_speech = max(max_speech, lo - prev_end)
            prev_end = max(prev_end, hi)
    # trailing speech after last silence
    if t_end > prev_end:
        max_speech = max(max_speech, t_end - prev_end)
    speech = max(0.0, duration - covered)
    return {
        "duration": round(duration, 3),
        "silence": round(covered, 3),
        "speech": round(speech, 3),
        "speech_ratio": round(speech / duration, 3) if duration > 0 else 0.0,
        "max_speech": round(max_speech, 3),
    }


def _is_suspicious_tail(stats: dict, min_speech: float = 1.5) -> bool:
    """True if tail has enough speech to suspect premature EOS."""
    return stats["speech"] >= min_speech or (
        stats["speech"] >= 1.0 and stats["speech_ratio"] >= 0.5
    )


def _is_suspicious_empty(stats: dict, min_speech: float = 2.0) -> bool:
    """True if an empty chunk has enough speech to suspect first-token EOS."""
    return stats["speech"] >= min_speech


# Tolerance for word-timestamp dedup at the rescue seam (seconds).
_RESCUE_DEDUP_TOL = 0.1

# --- Tail hallucination guard (v1) ---
# Layer 1: skip ASR entirely for tail chunks with negligible speech.
_TAIL_SKIP_SPEECH_SEC = 0.30
_TAIL_SKIP_MAX_SPEECH_SEC = 0.20
# Layer 2: after rescue, drop suspected hallucinated short text on low-speech tails.
_TAIL_HALLUCINATION_SPEECH_SEC = 0.40
_TAIL_HALLUCINATION_MAX_TEXT_LEN = 20


def _dedup_rescue_overlap(
    prev_text: str,
    rescue_text: str,
    rescue_words: list[WordTimestamp] | None,
    overlap_end: float,
) -> tuple[str, list[WordTimestamp]]:
    """Drop rescue content that overlaps already-transcribed audio.

    The 1s rescue overlap re-transcribes [last_char_end - 1s, last_char_end].
    Rescue words shifted to the chunk coordinate system whose start falls
    inside that overlap (start < overlap_end - tol) are duplicates and are
    dropped. Falls back to text-based strip_boundary when no word timestamps.
    """
    items = list(rescue_words or [])
    if items:
        kept = [w for w in items if float(w.start) >= overlap_end - _RESCUE_DEDUP_TOL]
        text = "".join(w.word for w in kept)
        return text, kept
    # No word timestamps: text-based fallback
    from ws_overlap import strip_boundary

    text = strip_boundary(prev_text, rescue_text, acoustic_overlap=True)
    return text, []


@dataclass
class OfflineTask:
    task_id: str
    filename: str
    audio_path: Path
    language: str | None
    status: str = "running"
    message: str = "0.00%"
    progress_percent: float = 0.0
    completed_windows: int = 0
    total_windows: int = 1
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    files: dict = field(default_factory=dict)
    result: dict | None = None

    def to_status(self) -> dict:
        out = {
            "task_id": self.task_id,
            "filename": self.filename,
            "status": self.status,
            "message": self.message,
            "progress_percent": self.progress_percent,
            "completed_windows": self.completed_windows,
            "total_windows": self.total_windows,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "files": self.files,
        }
        return out

    def submit_ack(self) -> dict:
        return {
            "ok": True,
            "task_id": self.task_id,
            "status": self.status,
            "progress_percent": self.progress_percent,
        }


class OfflineTaskStore:
    def __init__(self):
        self._tasks: dict[str, OfflineTask] = {}
        self._lock = asyncio.Lock()
        self._infer: InferFn | None = None

    def bind_infer(self, fn: InferFn) -> None:
        self._infer = fn

    async def _infer_split(
        self,
        audio: np.ndarray,
        sr: int,
        language: str | None,
        max_half: float = 20.0,
    ) -> tuple[str, list[WordTimestamp] | None]:
        """Split audio into ~max_half halves, infer each, merge text+words.

        Used as a second-level rescue when a full chunk returns empty.
        Changing the encoder's input (shorter segment) is more effective
        than changing the sampler seed/temperature for stubborn EOS.
        """
        total_dur = len(audio) / float(sr)
        if total_dur <= max_half * 1.5:
            # Already short enough — single inference
            chunk_size = max(FILE_CHUNK_SIZE_SEC, total_dur)
            result = await asyncio.wait_for(
                self._infer(audio, sr, language, "", chunk_size),
                timeout=REQUEST_TIMEOUT,
            )
            text = getattr(result, "text", "") or ""
            words = getattr(result, "words", None)
            return text, words

        mid = len(audio) // 2
        halves = [
            (audio[:mid], 0.0),
            (audio[mid:], mid / float(sr)),
        ]
        texts: list[str] = []
        all_words: list[WordTimestamp] = []
        for half_audio, offset in halves:
            half_dur = len(half_audio) / float(sr)
            half_size = max(FILE_CHUNK_SIZE_SEC, half_dur)
            result = await asyncio.wait_for(
                self._infer(half_audio, sr, language, "", half_size),
                timeout=REQUEST_TIMEOUT,
            )
            text = getattr(result, "text", "") or ""
            words = getattr(result, "words", None)
            if text:
                texts.append(text)
                if words:
                    all_words.extend(
                        WordTimestamp(
                            word=w.word,
                            start=round(float(w.start) + offset, 3),
                            end=round(float(w.end) + offset, 3),
                        )
                        for w in words
                    )
        merged = "".join(texts)
        return merged, (all_words if all_words else None)

    def has_active(self) -> bool:
        return any(t.status == "running" for t in self._tasks.values())

    def get(self, task_id: str) -> OfflineTask | None:
        return self._tasks.get(task_id)

    async def submit_path(self, audio_path: Path, language: str | None, filename: str | None = None) -> OfflineTask:
        task_id = _new_task_id()
        name = filename or audio_path.name
        task = OfflineTask(
            task_id=task_id,
            filename=name,
            audio_path=audio_path,
            language=language,
        )
        async with self._lock:
            self._tasks[task_id] = task
        asyncio.create_task(self._run(task))
        return task

    async def submit_upload(self, filename: str, data: bytes, language: str | None) -> OfflineTask:
        task_id = _new_task_id()
        name = Path(filename or "audio.wav").name
        out_dir = OUTPUT_DIR / task_id
        out_dir.mkdir(parents=True, exist_ok=True)
        audio_path = out_dir / name
        audio_path.write_bytes(data)
        task = OfflineTask(
            task_id=task_id,
            filename=name,
            audio_path=audio_path,
            language=language,
        )
        async with self._lock:
            self._tasks[task_id] = task
        asyncio.create_task(self._run(task))
        return task

    async def _run(self, task: OfflineTask) -> None:
        if self._infer is None:
            task.status = "failed"
            task.message = "infer function not bound"
            task.finished_at = time.time()
            return
        try:
            loop = asyncio.get_event_loop()
            audio, sr = await loop.run_in_executor(
                None, _decode_audio_file, task.audio_path
            )
            duration = float(audio.size) / float(sr) if audio.size else 0.0
            cfg = _offline_split_config()
            try:
                silences = await loop.run_in_executor(
                    None, scan_silences, audio, sr
                )
            except Exception as e:
                log.error("Offline VAD scan failed, falling back to hard cuts: {}", e)
                silences = []
            plans = plan_chunks(duration, silences, cfg)
            if not plans:
                task.total_windows = 1
                task.started_at = time.time()
                payload = {"segments": []}
            else:
                task.total_windows = len(plans)
                task.started_at = time.time()
                all_words = []
                texts: list[str] = []
                prev_text = ""
                for i, plan in enumerate(plans):
                    log.info(
                        "Offline task {} chunk {}/{} kind={} audio={:.3f}-{:.3f} drop_before={}",
                        task.task_id,
                        i + 1,
                        len(plans),
                        plan.kind,
                        plan.audio_start,
                        plan.audio_end,
                        plan.drop_before,
                    )
                    chunk = _slice_audio(audio, sr, plan.audio_start, plan.audio_end)
                    slice_dur = max(0.0, plan.audio_end - plan.audio_start)
                    chunk_size = max(FILE_CHUNK_SIZE_SEC, slice_dur)

                    # --- Layer 1: Tail VAD gate — skip ASR for negligible-speech tails
                    tail_pre_stats = None
                    if plan.kind == "tail":
                        tail_pre_stats = _speech_stats(
                            silences, plan.audio_start, plan.audio_end
                        )
                        if (
                            tail_pre_stats["speech"] < _TAIL_SKIP_SPEECH_SEC
                            and tail_pre_stats["max_speech"] < _TAIL_SKIP_MAX_SPEECH_SEC
                        ):
                            log.info(
                                "Offline coverage_guard | task={} chunk={}/{} "
                                "status=SKIP_TAIL_LOW_VAD "
                                "range={:.3f}-{:.3f} speech={:.3f} max_speech={:.3f}",
                                task.task_id, i + 1, len(plans),
                                plan.audio_start, plan.audio_end,
                                tail_pre_stats["speech"],
                                tail_pre_stats["max_speech"],
                            )
                            task.completed_windows = i + 1
                            pct = 100.0 * task.completed_windows / task.total_windows
                            task.progress_percent = round(pct, 2)
                            task.message = f"{task.progress_percent:.2f}%"
                            continue

                    result = await asyncio.wait_for(
                        self._infer(chunk, sr, task.language, "", chunk_size),
                        timeout=REQUEST_TIMEOUT,
                    )
                    raw_text = getattr(result, "text", "") or ""
                    n_gen = getattr(result, "n_generate", 0) or 0
                    result_words = getattr(result, "words", None)

                    # --- coverage_guard: detect premature EOS ---
                    guard_status = "ok"

                    if not raw_text:
                        chunk_stats = _speech_stats(
                            silences, plan.audio_start, plan.audio_end
                        )
                        if _is_suspicious_empty(chunk_stats):
                            guard_status = "SUSPICIOUS_EMPTY_EOS"
                            log.warning(
                                "Offline coverage_guard | task={} chunk={}/{} status={} "
                                "audio={:.3f}-{:.3f} raw_len=0 n_gen={} "
                                "vad_speech={:.3f}s vad_ratio={:.3f}",
                                task.task_id, i + 1, len(plans), guard_status,
                                plan.audio_start, plan.audio_end, n_gen,
                                chunk_stats["speech"], chunk_stats["speech_ratio"],
                            )
                            # Retry: same temp, new seed (sampler auto-randomizes)
                            retry_result = await asyncio.wait_for(
                                self._infer(chunk, sr, task.language, "", chunk_size),
                                timeout=REQUEST_TIMEOUT,
                            )
                            raw_text = getattr(retry_result, "text", "") or ""
                            n_gen = getattr(retry_result, "n_generate", 0) or 0
                            result_words = getattr(retry_result, "words", None)
                            if raw_text:
                                guard_status = "RESCUED_EMPTY_RETRY"
                                log.info(
                                    "Offline coverage_guard | task={} chunk={} status={} "
                                    "retry_raw_len={} retry_n_gen={}",
                                    task.task_id, i + 1, guard_status,
                                    len(raw_text), n_gen,
                                )
                            else:
                                guard_status = "EMPTY_RETRY_FAILED"
                                log.warning(
                                    "Offline coverage_guard | task={} chunk={} status={} "
                                    "retry still empty, trying split rescue",
                                    task.task_id, i + 1, guard_status,
                                )
                                # Level-2 rescue: split chunk into ~20s halves
                                split_text, split_words = await self._infer_split(
                                    chunk, sr, task.language
                                )
                                if split_text:
                                    raw_text = split_text
                                    result_words = split_words
                                    guard_status = "RESCUED_EMPTY_SPLIT"
                                    log.info(
                                        "Offline coverage_guard | task={} chunk={} status={} "
                                        "split_raw_len={}",
                                        task.task_id, i + 1, guard_status,
                                        len(raw_text),
                                    )
                                else:
                                    guard_status = "EMPTY_VAD_FALSE_POSITIVE"
                                    log.info(
                                        "Offline coverage_guard | task={} chunk={} status={} "
                                        "vad_speech={:.3f}s vad_ratio={:.3f} "
                                        "retry_raw=0 split_nonempty=0 action=accept_empty",
                                        task.task_id, i + 1, guard_status,
                                        chunk_stats["speech"], chunk_stats["speech_ratio"],
                                    )
                        else:
                            guard_status = "EMPTY_OK"

                    # Tail rescue: check for premature EOS after any non-empty result
                    # (including retry-rescued text). Recompute last_char_end from
                    # current result_words since retry may have replaced them.
                    if raw_text:
                        last_char_end = plan.audio_start
                        if result_words:
                            shifted_ends = [
                                plan.audio_start + float(w.end) for w in result_words
                            ]
                            if shifted_ends:
                                last_char_end = max(shifted_ends)
                        tail_stats = _speech_stats(
                            silences, last_char_end, plan.audio_end
                        )
                        if _is_suspicious_tail(tail_stats):
                            guard_status = "PREMATURE_EOS"
                            log.warning(
                                "Offline coverage_guard | task={} chunk={}/{} status={} "
                                "audio={:.3f}-{:.3f} raw_len={} n_gen={} "
                                "last_char_end={:.3f} tail_dur={:.3f} tail_speech={:.3f} tail_ratio={:.3f}",
                                task.task_id, i + 1, len(plans), guard_status,
                                plan.audio_start, plan.audio_end, len(raw_text), n_gen,
                                last_char_end, tail_stats["duration"],
                                tail_stats["speech"], tail_stats["speech_ratio"],
                            )
                            # Tail rescue: re-run only [last_char_end - 1s, chunk_end]
                            rescue_start = max(plan.audio_start, last_char_end - 1.0)
                            rescue_end = plan.audio_end
                            rescue_chunk = _slice_audio(audio, sr, rescue_start, rescue_end)
                            rescue_dur = max(0.0, rescue_end - rescue_start)
                            rescue_size = max(FILE_CHUNK_SIZE_SEC, rescue_dur)
                            rescue_result = await asyncio.wait_for(
                                self._infer(
                                    rescue_chunk, sr, task.language, "", rescue_size
                                ),
                                timeout=REQUEST_TIMEOUT,
                            )
                            rescue_text = getattr(rescue_result, "text", "") or ""
                            rescue_words = getattr(rescue_result, "words", None)
                            if rescue_text:
                                guard_status = "RESCUED_TAIL"
                                word_offset = rescue_start - plan.audio_start
                                shifted_rescue_words = [
                                    WordTimestamp(
                                        word=w.word,
                                        start=round(float(w.start) + word_offset, 3),
                                        end=round(float(w.end) + word_offset, 3),
                                    )
                                    for w in (rescue_words or [])
                                ]
                                rescue_text, kept_words = _dedup_rescue_overlap(
                                    raw_text, rescue_text, shifted_rescue_words,
                                    last_char_end,
                                )
                                log.info(
                                    "Offline coverage_guard | task={} chunk={} status={} "
                                    "rescue_range={:.3f}-{:.3f} rescue_raw_len={} "
                                    "dedup_kept={}",
                                    task.task_id, i + 1, guard_status,
                                    rescue_start, rescue_end, len(rescue_text),
                                    len(kept_words),
                                )
                                if rescue_text:
                                    raw_text = raw_text + rescue_text
                                    if result_words and kept_words:
                                        result_words = list(result_words) + kept_words
                                    elif kept_words:
                                        result_words = list(kept_words)
                                else:
                                    guard_status = "RESCUED_TAIL_OVERLAP_ONLY"
                                    log.info(
                                        "Offline coverage_guard | task={} chunk={} "
                                        "status={} action=rescue_fully_overlapped",
                                        task.task_id, i + 1, guard_status,
                                    )
                            else:
                                # Level-2 rescue: split tail into ~20s halves
                                split_text, split_words = await self._infer_split(
                                    rescue_chunk, sr, task.language
                                )
                                if split_text:
                                    guard_status = "RESCUED_TAIL_SPLIT"
                                    word_offset = rescue_start - plan.audio_start
                                    shifted_split_words = [
                                        WordTimestamp(
                                            word=w.word,
                                            start=round(float(w.start) + word_offset, 3),
                                            end=round(float(w.end) + word_offset, 3),
                                        )
                                        for w in (split_words or [])
                                    ]
                                    split_text, kept_words = _dedup_rescue_overlap(
                                        raw_text, split_text, shifted_split_words,
                                        last_char_end,
                                    )
                                    log.info(
                                        "Offline coverage_guard | task={} chunk={} status={} "
                                        "split_rescue_range={:.3f}-{:.3f} split_raw_len={} "
                                        "dedup_kept={}",
                                        task.task_id, i + 1, guard_status,
                                        rescue_start, rescue_end, len(split_text),
                                        len(kept_words),
                                    )
                                    if split_text:
                                        raw_text = raw_text + split_text
                                        if result_words and kept_words:
                                            result_words = list(result_words) + kept_words
                                        elif kept_words:
                                            result_words = list(kept_words)
                                    else:
                                        guard_status = "RESCUED_TAIL_SPLIT_OVERLAP_ONLY"
                                        log.info(
                                            "Offline coverage_guard | task={} chunk={} "
                                            "status={} action=split_rescue_fully_overlapped",
                                            task.task_id, i + 1, guard_status,
                                        )
                                else:
                                    guard_status = "TAIL_VAD_FALSE_POSITIVE"
                                    log.info(
                                        "Offline coverage_guard | task={} chunk={} "
                                        "status={} tail_speech={:.3f}s "
                                        "rescue_raw=0 split_nonempty=0 action=accept_tail_loss",
                                        task.task_id, i + 1, guard_status,
                                        tail_stats["speech"],
                                    )

                    # --- Layer 2: Tail hallucination guard (before stitch) ---
                    # If tail chunk had low speech, ASR produced short text, and
                    # rescue/split both added nothing, the text is likely a
                    # hallucination on noise/silence — drop it.
                    if (
                        plan.kind == "tail"
                        and tail_pre_stats is not None
                        and guard_status in (
                            "TAIL_VAD_FALSE_POSITIVE",
                            "RESCUED_TAIL_OVERLAP_ONLY",
                            "RESCUED_TAIL_SPLIT_OVERLAP_ONLY",
                        )
                        and tail_pre_stats["speech"] < _TAIL_HALLUCINATION_SPEECH_SEC
                        and len((raw_text or "").strip()) <= _TAIL_HALLUCINATION_MAX_TEXT_LEN
                    ):
                        log.warning(
                            "Offline coverage_guard | task={} chunk={}/{} "
                            "status=TAIL_HALLUCINATION_DROP "
                            "reason=premature_eos+empty_rescue+empty_split+low_vad+short_text "
                            "range={:.3f}-{:.3f} speech={:.3f} max_speech={:.3f} "
                            "raw={!r}",
                            task.task_id, i + 1, len(plans),
                            plan.audio_start, plan.audio_end,
                            tail_pre_stats["speech"],
                            tail_pre_stats["max_speech"],
                            raw_text,
                        )
                        raw_text = ""
                        result_words = []

                    text, words = finalize_offline_chunk(
                        raw_text,
                        result_words,
                        audio_start=plan.audio_start,
                        audio_end=plan.audio_end,
                        drop_before=plan.drop_before,
                        kind=plan.kind,
                        prev_text=prev_text,
                        prev_words=all_words,
                        eps=cfg.cut_eps,
                        overlap=cfg.hard_overlap,
                    )
                    if text != raw_text:
                        log.info(
                            "Offline boundary_stitch | task={} chunk={} raw={} out={}",
                            task.task_id,
                            i + 1,
                            raw_text[-24:] if raw_text else "",
                            text[-24:] if text else "",
                        )
                    if text:
                        texts.append(text)
                        all_words = append_offline_chunk(all_words, words)
                        prev_text = text
                    task.completed_windows = i + 1
                    pct = 100.0 * task.completed_windows / task.total_windows
                    task.progress_percent = round(pct, 2)
                    task.message = f"{task.progress_percent:.2f}%"
                full_text = ""
                for part in texts:
                    full_text = stitch_chunk_text(full_text, part)
                itn_text, itn_words = apply_chinese_itn(full_text, all_words)
                if itn_text != full_text:
                    log.info(
                        "Offline itn | task={} raw={} out={}",
                        task.task_id,
                        full_text[-32:] if full_text else "",
                        itn_text[-32:] if itn_text else "",
                    )
                    full_text = itn_text
                    all_words = itn_words
                # Hotword correction (phoneme-based, after ITN).
                hw_text, hw_words = apply_hotword(full_text, all_words)
                if hw_text != full_text:
                    log.info(
                        "Offline hotword | task={} raw={} out={}",
                        task.task_id,
                        full_text[-32:] if full_text else "",
                        hw_text[-32:] if hw_text else "",
                    )
                    full_text = hw_text
                    all_words = hw_words
                segments = words_to_segments(all_words or None, full_text)
                if ASR_REMOVE_FILLERS:
                    cleaned_segments = clean_filler_segments(segments)
                    if cleaned_segments != segments:
                        log.info(
                            "Offline fillers | task={} segments={}->{}",
                            task.task_id, len(segments), len(cleaned_segments),
                        )
                    segments = cleaned_segments
                payload = {"segments": segments}

            task.result = payload
            if OFFLINE_SAVE_RESULT:
                out_dir = OUTPUT_DIR / task.task_id
                out_dir.mkdir(parents=True, exist_ok=True)
                stem = Path(task.filename).stem or "audio"
                json_path = out_dir / f"{stem}.json"
                json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                task.files = {"json": str(json_path.resolve())}
                log.info("Offline task {} completed | windows={} json={}", task.task_id, task.total_windows, json_path)
            else:
                log.info("Offline task {} completed | windows={} (result not saved to disk)", task.task_id, task.total_windows)
            task.completed_windows = task.total_windows
            task.progress_percent = 100.0
            task.message = "100.00%"
            task.status = "completed"
            task.finished_at = time.time()
        except asyncio.TimeoutError:
            task.status = "failed"
            task.message = "Transcription timed out"
            task.finished_at = time.time()
            log.error("Offline task {} timed out", task.task_id)
        except Exception as e:
            task.status = "failed"
            task.message = str(e)
            task.finished_at = time.time()
            log.error("Offline task {} failed: {}", task.task_id, e)


offline_store = OfflineTaskStore()
