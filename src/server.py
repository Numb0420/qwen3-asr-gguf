from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
import uuid as _uuid_module
from contextlib import asynccontextmanager
from math import gcd
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from scipy.signal import butter, resample_poly, sosfilt

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
VENDOR = ROOT.parent / "vendor"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

from backend import QwenGGUFBackend
from backend.base import BackendState, WordTimestamp
from config import (
    ASR_USE_SERVER_VAD,
    ASR_REMOVE_FILLERS,
    ENABLE_ALIGNER,
    FORMAT_NUM,
    HOST,
    LLM_FN,
    LLM_USE_GPU,
    OFFLINE_CUT_EPS_SECONDS,
    ONNX_PROVIDER,
    PORT,
    REQUEST_TIMEOUT,
    TARGET_SR,
    UNLOAD_GRACE_S,
    apply_llm_device_env,
    WS_COMMITTED_SPEECH_SECONDS,
    WS_EARLY_SILENCE_MS,
    WS_FINAL_ALIGN,
    WS_FINAL_YIELD,
    WS_FIXED_ENCODER_WINDOW,
    WS_FLUSH_SILENCE_MS,
    WS_HARD_CUT_MIN_SILENCE_MS,
    WS_HARD_CUT_OVERLAP_SECONDS,
    WS_HARD_CUT_SEARCH_SECONDS,
    WS_LAZY_PARTIAL,
    WS_MAX_UTTERANCE_SECONDS,
    WS_PARTIAL_SECONDS,
    WS_PARTIAL_WINDOW_SEC,
    WS_SHARED_UTT,
    WS_SOFT_CUT_ENABLED,
    WS_SOFT_CUT_LOOKBACK_MS,
    WS_SOFT_CUT_SILENCE_MS,
    WS_SOFT_CUT_START_SECONDS,
    WS_VAD_HANGOVER_MS,
    WS_VAD_PRE_ROLL_MS,
    WS_VAD_END_MAX_SILENCE_MS,
    validate_env,
)
from errors import error_response
from hotword_correct import apply_hotword
from inference_queue import infer_queue
from itn import apply_chinese_itn
from logger import log, reset_request_id, set_request_id
from offline_tasks import offline_store
from repetition import carry_prefix_tail, collapse_repetitions
from schemas import (
    API_DESCRIPTION,
    API_TAGS,
    ErrorResponse,
    HealthResponse,
    TaskResultResponse,
    TaskStatusResponse,
    TaskSubmitResponse,
    TranscribePathRequest,
)
from vad import (
    ENERGY_ONSET_RMS,
    RealtimeVad,
    advance_energy_run,
    choose_hard_cut_silence,
    energy_onset_ready,
    scan_silences,
    silence_threshold_samples,
)
from fillers import clean_fillers, clean_filler_segments
from ws_overlap import (
    OrderedFinalBuffer,
    PartialRequestGate,
    append_pre_roll,
    apply_hard_cut,
    fallback_empty_final,
    fresh_audio_samples,
    join_partial_tail,
    pending_caption_prefix,
    plan_partial_window,
    plan_soft_cut,
    split_soft_cut,
    finalize_ws_utterance,
    leftover_echo,
    discard_stale_overlap_after_skip,
    retie_turn_overlap,
    stitch_prev_text,
    strip_partial_terminal_punct,
    strip_boundary,
    strip_forced_cut_stop,
    sync_words_to_text,
    text_ends_sentence,
    unused_pre_roll,
)

backend = QwenGGUFBackend()
_last_used = 0.0
_active_ws: WebSocket | None = None
_ws_lock = asyncio.Lock()


def _touch() -> None:
    global _last_used
    _last_used = time.time()


def _telephony_bandpass(audio: np.ndarray, sr: int) -> np.ndarray:
    sos = butter(4, [300, 3400], btype="bandpass", fs=sr, output="sos")
    return sosfilt(sos, audio).astype(np.float32)


def _resample_pcm_bytes(pcm_bytes: bytes, orig_sr: int) -> bytes:
    if orig_sr == TARGET_SR:
        return pcm_bytes
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    g = gcd(orig_sr, TARGET_SR)
    resampled = resample_poly(samples, TARGET_SR // g, orig_sr // g)
    return resampled.astype(np.int16).tobytes()


def detect_and_fix_repetitions(text: str, max_repeats: int = 2) -> str:
    # Official-style Chinese-safe collapse (threshold 10). max_repeats kept for call-site compatibility.
    return collapse_repetitions(text)


def _ws_preview(text: str, limit: int = 48) -> str:
    raw = (text or "").replace("\n", " ")
    if len(raw) <= limit:
        return f"n={len(raw)} {raw!r}"
    return f"n={len(raw)} {raw[:28]!r}…{raw[-16:]!r}"


def _pcm_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


async def _ensure_model_loaded() -> None:
    deadline = time.time() + 60.0
    waited = False
    while True:
        if backend.loaded:
            _touch()
            return
        if backend.state in (BackendState.UNLOADING, BackendState.LOADING):
            if time.time() >= deadline:
                raise RuntimeError(
                    f"Timed out waiting for backend {backend.state.value}"
                )
            if not waited:
                log.info("Waiting for backend {} before load", backend.state.value)
                waited = True
            await asyncio.sleep(0.05)
            continue
        break
    await infer_queue.run_blocking(backend.load)
    _touch()


async def _unload_model(
    *,
    reason: str = "unspecified",
    force: bool = False,
    ignore_ws: bool = False,
) -> None:
    await infer_queue.wait_idle()
    if not force:
        if not ignore_ws and _active_ws is not None:
            log.info("Skip unload | reason={} blocked_by=live WebSocket", reason)
            return
        if offline_store.has_active():
            log.info("Skip unload | reason={} blocked_by=offline task running", reason)
            return
    log.info("Unloading model | reason={} force={}", reason, force)
    if backend.loaded or backend.state != BackendState.UNLOADED:
        await infer_queue.run_blocking(backend.unload)


async def _run_asr(
    audio: np.ndarray,
    language: str | None,
    *,
    realtime: bool,
    is_final: bool,
    priority: int,
    kind: str,
    session_id: str = "",
    utt_id: int = 0,
    prefix_text: str = "",
    timing_out: dict[str, float] | None = None,
    audio_start_sample: int = 0,
    use_utt_cache: bool = True,
    origin_audio: np.ndarray | None = None,
    origin_start_sample: int | None = None,
    yield_event: threading.Event | None = None,
    allow_final_yield: bool = True,
):
    await _ensure_model_loaded()
    _touch()

    # 每个 job 独立 abort_event：final 入队时 infer_queue 会 set 同 utt 正在跑的
    # partial 的 event，partial 的 decode 循环命中后尽快退出，腾出 worker 给 final。
    abort_event = threading.Event()
    if yield_event is None and allow_final_yield:
        yield_event = threading.Event()
    submitted_at = time.perf_counter()

    def _job():
        started_at = time.perf_counter()
        if timing_out is not None:
            timing_out["queue_ms"] = (started_at - submitted_at) * 1000.0
        try:
            if realtime:
                return backend.transcribe_realtime(
                    audio,
                    TARGET_SR,
                    language,
                    is_final=is_final,
                    prefix_text=prefix_text,
                    abort_event=abort_event,
                    session_id=session_id,
                    utt_id=utt_id,
                    audio_start_sample=audio_start_sample,
                    use_utt_cache=use_utt_cache,
                    origin_audio=origin_audio,
                    origin_start_sample=origin_start_sample,
                    yield_event=yield_event,
                )
            return backend.transcribe_file(audio, TARGET_SR, language)
        finally:
            if timing_out is not None:
                timing_out["infer_ms"] = (time.perf_counter() - started_at) * 1000.0

    result = await asyncio.wait_for(
        infer_queue.submit(
            _job,
            priority=priority,
            kind=kind,
            session_id=session_id if realtime else "",
            utt_id=utt_id if realtime else 0,
            abort_event=abort_event,
            yield_event=yield_event,
        ),
        timeout=REQUEST_TIMEOUT,
    )
    if result is None:
        return None
    result.text = detect_and_fix_repetitions(result.text)
    return result


async def _unload_watchdog() -> None:
    while True:
        await asyncio.sleep(15)
        if UNLOAD_GRACE_S <= 0:
            continue
        if _active_ws is not None:
            continue
        if offline_store.has_active():
            continue
        if not backend.loaded:
            continue
        if time.time() - _last_used <= UNLOAD_GRACE_S:
            continue
        log.info("Unload grace reached ({}s), unloading model", UNLOAD_GRACE_S)
        try:
            await _unload_model(reason="idle_grace")
        except Exception as e:
            log.error("Idle unload failed: {}", e)


@asynccontextmanager
async def lifespan(_app):
    apply_llm_device_env()
    validate_env()
    infer_queue.start()
    async def _offline_infer(audio, sr, language, prefix_text="", chunk_size_sec=None):
        await _ensure_model_loaded()
        _touch()

        def _job():
            return backend.transcribe_file(
                audio,
                sr,
                language,
                prefix_text=prefix_text,
                chunk_size_sec=chunk_size_sec,
            )

        result = await infer_queue.submit(_job, priority=1, kind="normal")
        if result is None:
            raise RuntimeError("Inference result was dropped")
        result.text = detect_and_fix_repetitions(result.text)
        return result

    offline_store.bind_infer(_offline_infer)
    try:
        await _ensure_model_loaded()
    except Exception as e:
        log.error("Startup model load failed: {}", e)
    watchdog = asyncio.create_task(_unload_watchdog())
    yield
    watchdog.cancel()
    try:
        await _unload_model(reason="shutdown", force=True)
    except Exception:
        pass
    infer_queue.stop()


_uvicorn_server = None


def bind_uvicorn_server(server) -> None:
    """Allow /admin/shutdown to stop the uvicorn Server started from run.py."""
    global _uvicorn_server
    _uvicorn_server = server


app = FastAPI(
    title="Qwen3-ASR",
    version="1.0.0",
    description=API_DESCRIPTION,
    openapi_tags=API_TAGS,
    lifespan=lifespan,
    responses={
        422: {"model": ErrorResponse, "description": "Audio decode or validation error"},
        504: {"model": ErrorResponse, "description": "Inference timed out"},
    },
)


@app.middleware("http")
async def _request_id_middleware(request, call_next):
    req_id = request.headers.get("x-request-id") or str(_uuid_module.uuid4())
    token = set_request_id(req_id)
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response
    finally:
        reset_request_id(token)


@app.get("/health", response_model=HealthResponse, tags=["System"], summary="Health check")
async def health():
    return {
        "status": "ok",
        "model_loaded": backend.loaded,
        "backend": "gguf",
        "state": backend.state.value,
        "model_id": LLM_FN,
        "onnx_provider": ONNX_PROVIDER,
        "llm_gpu": "vulkan" if LLM_USE_GPU else "cpu",
        "aligner": ENABLE_ALIGNER,
        "ws_final_align": WS_FINAL_ALIGN,
        "format_num": FORMAT_NUM,
    }


@app.post("/admin/warmup", tags=["System"], summary="Load the ASR model")
async def admin_warmup():
    await _ensure_model_loaded()
    return {
        "ok": True,
        "state": backend.state.value,
        "model_loaded": backend.loaded,
    }


@app.post("/admin/shutdown", tags=["System"], summary="Stop the ASR server")
async def admin_shutdown():
    if _uvicorn_server is not None:
        _uvicorn_server.should_exit = True
        _uvicorn_server.force_exit = True
    return {"ok": True}


@app.post(
    "/offline/transcribe",
    response_model=TaskSubmitResponse,
    tags=["Transcription"],
    summary="Upload audio and start an offline task",
)
async def transcribe_upload(
    file: UploadFile = File(..., description="Audio file (WAV, FLAC, MP3, OGG, etc.)"),
    language: str = Form("auto", description="Language code: zh / en / auto"),
):
    data = await file.read()
    if not data:
        return error_response("AUDIO_DECODE_FAILED", "Empty upload", 422)
    lang = None if language == "auto" else language
    log.info("POST /offline/transcribe | file={} size={} language={}", file.filename, len(data), language)
    task = await offline_store.submit_upload(file.filename or "audio.wav", data, lang)
    return task.submit_ack()


@app.post(
    "/offline/transcribe-path",
    response_model=TaskSubmitResponse,
    tags=["Transcription"],
    summary="Transcribe a local audio path",
)
async def transcribe_path(body: TranscribePathRequest):
    audio_path = Path(body.audio_path).expanduser()
    if not audio_path.is_file():
        return error_response("AUDIO_NOT_FOUND", f"wav not found: {audio_path}", 422)
    lang = None if body.language == "auto" else body.language
    log.info("POST /offline/transcribe-path | path={} language={}", audio_path, body.language)
    task = await offline_store.submit_path(audio_path, lang)
    return task.submit_ack()


@app.get(
    "/offline/tasks/{task_id}",
    response_model=TaskStatusResponse,
    tags=["Transcription"],
    summary="Offline task status",
)
async def get_task(task_id: str):
    task = offline_store.get(task_id)
    if task is None:
        return error_response("TASK_NOT_FOUND", f"Unknown task_id: {task_id}", 404)
    return task.to_status()


@app.get(
    "/offline/tasks/{task_id}/result",
    response_model=TaskResultResponse,
    tags=["Transcription"],
    summary="Offline task result",
)
async def get_task_result(task_id: str):
    task = offline_store.get(task_id)
    if task is None:
        return error_response("TASK_NOT_FOUND", f"Unknown task_id: {task_id}", 404)
    if task.status != "completed" or task.result is None:
        return error_response("TASK_NOT_READY", task.message or task.status, 409)
    return task.result


@app.websocket("/realtime/stream")
async def websocket_transcribe(websocket: WebSocket):
    global _active_ws

    async with _ws_lock:
        if _active_ws is not None:
            await websocket.accept()
            await websocket.send_json({
                "code": "WS_BUSY",
                "message": "Only one WebSocket transcription session is allowed",
                "statusCode": 409,
            })
            await websocket.close(code=1008)
            return
        await websocket.accept()
        _active_ws = websocket

    ws_req_id = websocket.query_params.get("request_id") or str(_uuid_module.uuid4())
    token = set_request_id(ws_req_id)

    use_vad = ASR_USE_SERVER_VAD
    vad_param = websocket.query_params.get("use_server_vad")
    if vad_param is not None:
        use_vad = vad_param.lower() in ("true", "1", "yes")

    client_sr = int(websocket.query_params.get("sample_rate", str(TARGET_SR)))
    if client_sr not in (8000, 16000):
        await websocket.send_json({
            "code": "UNSUPPORTED_SAMPLE_RATE",
            "message": f"sample_rate must be 8000 or 16000, got {client_sr}",
            "statusCode": 400,
        })
        await websocket.close()
        async with _ws_lock:
            if _active_ws is websocket:
                _active_ws = None
        reset_request_id(token)
        return

    log.info(
        "[WS] Client connected | request_id={} use_vad={} sample_rate={}",
        ws_req_id,
        use_vad,
        client_sr,
    )

    lang_code: str | None = None
    accept_pcm = True
    speech_buf = np.zeros(0, dtype=np.float32)
    pre_roll = np.zeros(0, dtype=np.float32)
    last_partial_at = 0
    last_partial_by_utt: dict[int, str] = {}
    last_partial_vad_by_utt: dict[int, str] = {}
    last_partial_audio_end: dict[int, float] = {}
    partial_block_start: dict[int, float] = {}
    partial_committed_text: dict[int, str] = {}
    partial_committed_vad: dict[int, str] = {}
    partial_gate = PartialRequestGate()
    final_buffer = OrderedFinalBuffer()
    last_emitted_by_utt: dict[int, str] = {}
    finalized_utts: set[int] = set()
    final_submitted_at: float | None = None
    final_submit_utt = -1
    endpoint_wait_started = 0.0
    lazy_partial_pending = False
    stream_samples = 0
    skip_samples = 0
    skip_from = 0.0
    last_skip_log = 0
    hangover_left = 0
    hangover_heard = False
    energy_run = 0
    energy_hold = False
    energy_silence = 0
    punct_defer = False
    punct_defer_silence = 0
    buffer_audio_start = 0.0
    commit_start = 0.0
    drop_before: float | None = None
    utt_id = 0
    carry_prefix = ""
    speculative_prefix = ""
    carry_from_utt = -1
    turn_overlap = np.zeros(0, dtype=np.float32)
    send_lock = asyncio.Lock()
    inflight: set[asyncio.Task] = set()
    vad = RealtimeVad()
    _flush_lazy_cb = None
    partial_samples = int(WS_PARTIAL_SECONDS * TARGET_SR)
    max_samples = int(WS_MAX_UTTERANCE_SECONDS * TARGET_SR)
    pre_roll_samples = int(round(WS_VAD_PRE_ROLL_MS * TARGET_SR / 1000.0))
    hangover_samples = int(round(WS_VAD_HANGOVER_MS * TARGET_SR / 1000.0))
    punct_defer_cap = int(round(WS_VAD_END_MAX_SILENCE_MS * TARGET_SR / 1000.0))

    try:
        await _ensure_model_loaded()
        await websocket.send_json({
            "status": "connected",
            "sample_rate": client_sr,
            "format": "pcm_s16le",
            "partial_seconds": WS_PARTIAL_SECONDS,
            "partial_window_sec": WS_PARTIAL_WINDOW_SEC,
            "max_utterance_seconds": WS_MAX_UTTERANCE_SECONDS,
            "hard_cut_search_seconds": WS_HARD_CUT_SEARCH_SECONDS,
            "hard_cut_min_silence_ms": WS_HARD_CUT_MIN_SILENCE_MS,
            "hard_cut_overlap_seconds": WS_HARD_CUT_OVERLAP_SECONDS,
            "flush_silence_ms": WS_FLUSH_SILENCE_MS,
            "early_silence_ms": WS_EARLY_SILENCE_MS,
            "vad_pre_roll_ms": WS_VAD_PRE_ROLL_MS,
            "vad_hangover_ms": WS_VAD_HANGOVER_MS,
            "vad_end_max_silence_ms": WS_VAD_END_MAX_SILENCE_MS,
            "use_server_vad": use_vad,
            "ws_final_align": WS_FINAL_ALIGN,
            "remove_fillers": ASR_REMOVE_FILLERS,
            "fixed_encoder_window": WS_FIXED_ENCODER_WINDOW,
            "shared_utt": WS_SHARED_UTT,
            "lazy_partial": WS_LAZY_PARTIAL,
            "final_yield": WS_FINAL_YIELD,
            "soft_cut_enabled": WS_SOFT_CUT_ENABLED,
            "soft_cut_start_seconds": WS_SOFT_CUT_START_SECONDS,
            "soft_cut_silence_ms": WS_SOFT_CUT_SILENCE_MS,
            "soft_cut_lookback_ms": WS_SOFT_CUT_LOOKBACK_MS,
        })

        async def _send(payload: dict) -> None:
            async with send_lock:
                await websocket.send_json(payload)

        async def _send_final_ordered(job_utt_id: int, payload: dict | None) -> None:
            async with send_lock:
                ready = final_buffer.put(job_utt_id, payload)
                if not ready:
                    log.info("WS final_buffered | utt={} waiting_for={}",
                             job_utt_id, final_buffer.next_utt_id)
                for ready_utt, ready_payload in ready:
                    if ready_payload is not None:
                        await websocket.send_json(ready_payload)
                        log.info("WS final_sent | utt={} start={} end={}", ready_utt,
                                 ready_payload.get("start"), ready_payload.get("end"))
                    finalized_utts.add(ready_utt)
                if len(finalized_utts) > 16:
                    finalized_utts.difference_update(
                        uid for uid in list(finalized_utts) if uid < final_buffer.next_utt_id - 8
                    )

        def _stream_cursor() -> float:
            return stream_samples / TARGET_SR

        def _begin_next_utterance(reason: str, *, reset_vad: bool = True, hangover: bool = False) -> None:
            nonlocal speech_buf, last_partial_at, utt_id, turn_overlap
            nonlocal buffer_audio_start, commit_start, drop_before
            nonlocal hangover_left, hangover_heard
            nonlocal punct_defer, punct_defer_silence
            nonlocal energy_run, energy_hold, energy_silence
            cleared = speech_buf.size / TARGET_SR if speech_buf.size else 0.0
            old_utt = utt_id
            speech_buf = np.zeros(0, dtype=np.float32)
            turn_overlap = np.zeros(0, dtype=np.float32)
            last_partial_at = 0
            utt_id += 1
            if reset_vad:
                vad.reset()
            hangover_heard = False
            hangover_left = hangover_samples if hangover and hangover_samples > 0 else 0
            punct_defer = False
            punct_defer_silence = 0
            energy_run = 0
            energy_hold = False
            energy_silence = 0
            buffer_audio_start = _stream_cursor()
            commit_start = buffer_audio_start
            drop_before = None
            log.info(
                "WS turn_clear | reason={} utt={}→{} cursor={:.3f} cleared={:.3f}s "
                "vad_reset={} hangover={:.3f}s",
                reason,
                old_utt,
                utt_id,
                buffer_audio_start,
                cleared,
                int(reset_vad),
                hangover_left / TARGET_SR,
            )

        def _apply_hard_cut(cut_plan=None) -> np.ndarray | None:
            nonlocal speech_buf, last_partial_at, utt_id, turn_overlap
            nonlocal buffer_audio_start, commit_start, drop_before
            nonlocal punct_defer, punct_defer_silence
            nonlocal pre_roll
            before_start = buffer_audio_start
            before_dur = speech_buf.size / TARGET_SR if speech_buf.size else 0.0
            old_utt = utt_id
            previous_audio = None
            if cut_plan is not None:
                previous_audio, state = split_soft_cut(
                    speech_buf,
                    buffer_audio_start,
                    cut_plan,
                    TARGET_SR,
                )
            else:
                state = apply_hard_cut(
                    speech_buf,
                    buffer_audio_start,
                    WS_HARD_CUT_OVERLAP_SECONDS,
                    TARGET_SR,
                    0.0,
                )
            speech_buf = state.speech_buf
            turn_overlap = np.zeros(0, dtype=np.float32)
            pre_roll = np.zeros(0, dtype=np.float32)
            buffer_audio_start = state.buffer_audio_start
            commit_start = state.commit_start
            drop_before = state.drop_before
            last_partial_at = state.last_partial_at
            punct_defer = False
            punct_defer_silence = 0
            utt_id += 1
            log.info(
                "WS hard_cut | utt={}→{} mode={} sent={:.3f}-{:.3f} keep={:.3f}s "
                "audio_start={:.3f}→{:.3f} commit={:.3f} drop_before={:.3f} vad_reset=0",
                old_utt,
                utt_id,
                "vad_silence" if cut_plan is not None else "fallback",
                before_start,
                cut_plan.previous_end if cut_plan is not None else before_start + before_dur,
                speech_buf.size / TARGET_SR,
                before_start,
                buffer_audio_start,
                commit_start,
                state.drop_before,
            )
            return previous_audio

        def _apply_soft_cut(plan) -> np.ndarray:
            nonlocal speech_buf, last_partial_at, utt_id, turn_overlap
            nonlocal buffer_audio_start, commit_start, drop_before
            nonlocal punct_defer, punct_defer_silence
            before_start = buffer_audio_start
            before_dur = speech_buf.size / TARGET_SR if speech_buf.size else 0.0
            old_utt = utt_id
            previous_audio, state = split_soft_cut(
                speech_buf,
                buffer_audio_start,
                plan,
                TARGET_SR,
            )
            speech_buf = state.speech_buf
            turn_overlap = np.zeros(0, dtype=np.float32)
            buffer_audio_start = state.buffer_audio_start
            commit_start = state.commit_start
            drop_before = state.drop_before
            last_partial_at = state.last_partial_at
            punct_defer = False
            punct_defer_silence = 0
            utt_id += 1
            log.info(
                "WS soft_cut | utt={}→{} sent={:.3f}-{:.3f} previous_end={:.3f} "
                "next_start={:.3f} silence_start={:.3f} overlap={:.3f}s "
                "audio_start={:.3f}→{:.3f} commit={:.3f} drop_before={:.3f} "
                "keep={:.3f}s vad_reset=0",
                old_utt,
                utt_id,
                before_start,
                before_start + before_dur,
                plan.previous_end,
                plan.next_start,
                plan.silence_start,
                plan.overlap_seconds,
                before_start,
                buffer_audio_start,
                commit_start,
                drop_before,
                speech_buf.size / TARGET_SR,
            )
            return previous_audio

        def _apply_vad_turn() -> None:
            nonlocal speech_buf, last_partial_at, utt_id, turn_overlap, pre_roll
            nonlocal buffer_audio_start, commit_start, drop_before
            nonlocal hangover_left, hangover_heard
            nonlocal punct_defer, punct_defer_silence
            nonlocal energy_run, energy_hold, energy_silence
            before_start = buffer_audio_start
            before_dur = speech_buf.size / TARGET_SR if speech_buf.size else 0.0
            old_utt = utt_id
            speech_buf = np.zeros(0, dtype=np.float32)
            turn_overlap = np.zeros(0, dtype=np.float32)
            pre_roll = np.zeros(0, dtype=np.float32)
            buffer_audio_start = _stream_cursor()
            commit_start = buffer_audio_start
            drop_before = None
            last_partial_at = 0
            hangover_heard = False
            hangover_left = hangover_samples if hangover_samples > 0 else 0
            punct_defer = False
            punct_defer_silence = 0
            energy_run = 0
            energy_hold = False
            energy_silence = 0
            utt_id += 1
            log.info(
                "WS vad_turn | utt={}→{} sent={:.3f}-{:.3f} overlap=0.000s "
                "audio_start={:.3f} commit={:.3f} drop_before={} hangover={:.3f}s vad_reset=0",
                old_utt,
                utt_id,
                before_start,
                before_start + before_dur,
                buffer_audio_start,
                commit_start,
                drop_before,
                hangover_left / TARGET_SR,
            )

        async def _run_and_send(
            job_utt_id: int,
            is_final: bool,
            audio: np.ndarray,
            audio_start: float,
            job_commit_start: float,
            job_drop_before: float | None,
            prefix: str,
            hard_cut: bool,
            reason: str,
            tail_partial: bool = False,
            partial_block_complete: bool = False,
            origin_audio: np.ndarray | None = None,
            origin_start: float | None = None,
            endpoint_wait_ms: float = 0.0,
        ) -> None:
            nonlocal carry_prefix, speculative_prefix, carry_from_utt, last_partial_by_utt
            nonlocal last_partial_audio_end
            nonlocal last_emitted_by_utt, finalized_utts
            nonlocal final_submitted_at, final_submit_utt
            raw_for_carry = ""
            kind = "final" if is_final else "partial"
            t0 = time.perf_counter()
            timing: dict[str, float] = {}
            origin_start_sample = (
                int(round(float(origin_start) * TARGET_SR))
                if origin_start is not None
                else int(round(float(audio_start) * TARGET_SR))
            )
            result = None
            while True:
                result = await _run_asr(
                    audio,
                    lang_code,
                    realtime=True,
                    is_final=is_final,
                    priority=0,
                    kind=kind,
                    session_id=ws_req_id,
                    utt_id=job_utt_id,
                    prefix_text="",
                    timing_out=timing,
                    audio_start_sample=int(round(float(audio_start) * TARGET_SR)),
                    use_utt_cache=not tail_partial,
                    origin_audio=origin_audio,
                    origin_start_sample=origin_start_sample,
                    allow_final_yield=not (is_final and hard_cut),
                )
                yielded = bool(result is not None and (getattr(result, "perf", None) or {}).get("yielded"))
                if is_final and yielded and WS_FINAL_YIELD:
                    log.info("WS final_yield | utt={} next_chunk={}", job_utt_id, (result.perf or {}).get("next_chunk"))
                    continue
                break
            infer_done_at = time.perf_counter()

            def _release_final_cache() -> None:
                # Cut already advanced utt_id. Drop only after this final has
                # read the bucket; a coalesced job never ran and must not wipe it.
                if is_final and result is not None:
                    backend.drop_utt_cache(ws_req_id, job_utt_id)

            waited = time.perf_counter() - t0
            duration = float(audio.size) / float(TARGET_SR) if audio.size else 0.0
            audio_end = audio_start + duration
            rtf = (waited / duration) if duration > 0 else 0.0
            perf = getattr(result, "perf", None) or {}
            log.info(
                "WS timing | kind={} reason={} utt={} audio_end={:.3f} "
                "audio_dur={:.3f}s total_ms={:.1f} queue_ms={:.1f} infer_ms={:.1f} "
                "encode_ms={} prefill_ms={} decode_ms={} n_chunks={} "
                "cache_hits={} decode_skipped={} result={}",
                kind,
                reason,
                job_utt_id,
                audio_end,
                duration,
                waited * 1000.0,
                timing.get("queue_ms", -1.0),
                timing.get("infer_ms", -1.0),
                perf.get("encode_ms", -1),
                perf.get("prefill_ms", -1),
                perf.get("decode_ms", -1),
                perf.get("n_chunks", -1),
                perf.get("cache_hits", 0),
                perf.get("decode_skipped", 0),
                "coalesced" if result is None else "ready",
            )
            log.info(
                "WS clocks | kind={} reason={} utt={} infer_done=1 ws_emit=0 "
                "text_changed=-1 endpoint_wait_ms={:.1f} infer_done_ms={:.1f}",
                kind,
                reason,
                job_utt_id,
                endpoint_wait_ms,
                (infer_done_at - t0) * 1000.0,
            )
            if carry_from_utt == job_utt_id - 1:
                # 只有存在真实音频重叠时才用 carry_prefix 做文本去重。
                # audio_start < commit_start 表示新句音频包含了上一句的 overlap 音频；
                # overlap=0 时 audio_start == commit_start，此时 carry_prefix 与新句
                # 文本中的相同字符纯属巧合（如"中"vs"中国共产党"），去重会误删。
                if audio_start < job_commit_start:
                    stitch_src = carry_prefix or prefix
                else:
                    stitch_src = ""
            else:
                stitch_src = prefix or carry_prefix
            if result is None:
                log.warning(
                    "WS drop | reason=queue_coalesce kind={} utt={} audio={:.3f}-{:.3f} "
                    "wait={:.2f}s rtf={:.2f} prefix={}",
                    kind,
                    job_utt_id,
                    audio_start,
                    audio_end,
                    waited,
                    rtf,
                    _ws_preview(prefix),
                )
                if is_final:
                    await _send_final_ordered(job_utt_id, None)
                return
            if not is_final and job_utt_id != utt_id:
                log.info(
                    "WS drop | reason=stale_utt kind=partial job_utt={} current_utt={} "
                    "audio={:.3f}-{:.3f} raw={}",
                    job_utt_id,
                    utt_id,
                    audio_start,
                    audio_end,
                    _ws_preview(result.text or ""),
                )
                return
            raw_text = result.text or ""
            log.info(
                "WS asr_raw | kind={} reason={} utt={} audio={:.3f}-{:.3f} "
                "wait+decode={:.2f}s rtf={:.2f} aligner_chars={} drop_before={} "
                "commit={:.3f} submit_prefix={} stitch={} raw={}",
                kind,
                reason,
                job_utt_id,
                audio_start,
                audio_end,
                waited,
                rtf,
                len(result.words or []),
                job_drop_before,
                job_commit_start,
                _ws_preview(prefix),
                _ws_preview(stitch_src),
                _ws_preview(raw_text),
            )
            text = raw_text
            if is_final:
                text, start, end, words, overlap = finalize_ws_utterance(
                    result.words,
                    text,
                    audio_start=audio_start,
                    audio_duration=duration,
                    commit_start=job_commit_start,
                    drop_before=job_drop_before,
                    eps=OFFLINE_CUT_EPS_SECONDS,
                    overlap_prefix=stitch_src,
                )
                if overlap.match_n:
                    log.info(
                        "WS overlap_match | utt={} n={} text={}",
                        job_utt_id,
                        overlap.match_n,
                        _ws_preview(overlap.match_plain),
                    )
                if overlap.contact_old is not None:
                    log.info(
                        "WS overlap_time_confirm | utt={} contact_old={}",
                        job_utt_id,
                        int(overlap.contact_old),
                    )
                if overlap.removed_plain:
                    log.info(
                        "WS overlap_actual_remove | utt={} text={}",
                        job_utt_id,
                        _ws_preview(overlap.removed_plain),
                    )
                if hard_cut:
                    cut_stop = strip_forced_cut_stop(text)
                    if cut_stop != text:
                        log.info(
                            "WS hard_cut_stop | utt={} raw={} out={}",
                            job_utt_id,
                            _ws_preview(text),
                            _ws_preview(cut_stop),
                        )
                        words = sync_words_to_text(words, cut_stop)
                        text = cut_stop
                        if words:
                            start = max(float(job_commit_start), float(words[0].start))
                            end = float(words[-1].end)
                elif reason == "soft_cut":
                    # Decoder may add a fake trailing "。" on a dangling last char.
                    # Strip it so the next overlapping window can match the leftover.
                    cut_stop = strip_forced_cut_stop(text)
                    if cut_stop != text:
                        log.info(
                            "WS soft_cut_stop | utt={} raw={} out={}",
                            job_utt_id,
                            _ws_preview(text),
                            _ws_preview(cut_stop),
                        )
                        words = sync_words_to_text(words, cut_stop)
                        text = cut_stop
                        if words:
                            start = max(float(job_commit_start), float(words[0].start))
                            end = float(words[-1].end)
                if raw_text and not text:
                    log.warning(
                        "WS drop | reason=finalize_empty utt={} raw={} prefix={}",
                        job_utt_id,
                        _ws_preview(raw_text),
                        _ws_preview(stitch_src),
                    )
                if not text:
                    fallback = fallback_empty_final(
                        raw_text,
                        text,
                        last_partial_by_utt.get(job_utt_id, ""),
                        overlap_prefix=stitch_src,
                        drop_before=job_drop_before,
                    )
                    if fallback and leftover_echo(stitch_src, fallback):
                        fallback = ""
                    if fallback:
                        text = fallback
                        log.warning(
                            "WS fallback | empty_final used last_partial utt={} out={}",
                            job_utt_id,
                            _ws_preview(text),
                        )
                if leftover_echo(stitch_src, text):
                    log.info(
                        "WS drop | reason=overlap_echo utt={} raw={} prefix={} out={}",
                        job_utt_id,
                        _ws_preview(raw_text),
                        _ws_preview(stitch_src),
                        _ws_preview(text),
                    )
                    _release_final_cache()
                    await _send_final_ordered(job_utt_id, None)
                    return
                raw_for_carry = text
                itn_text, itn_words = apply_chinese_itn(text, words)
                if itn_text != text:
                    log.info(
                        "WS itn | utt={} raw={} out={}",
                        job_utt_id,
                        _ws_preview(text),
                        _ws_preview(itn_text),
                    )
                    text = itn_text
                    words = itn_words
                    if words:
                        start = max(float(job_commit_start), float(words[0].start))
                        end = float(words[-1].end)
                # Hotword correction (phoneme-based, after ITN).
                hw_text, hw_words = apply_hotword(text, words)
                if hw_text != text:
                    log.info(
                        "WS hotword | utt={} raw={} out={}",
                        job_utt_id,
                        _ws_preview(text),
                        _ws_preview(hw_text),
                    )
                    text = hw_text
                    words = hw_words
                    if words:
                        start = max(float(job_commit_start), float(words[0].start))
                        end = float(words[-1].end)
                if ASR_REMOVE_FILLERS:
                    cleaned = clean_fillers(text)
                    if cleaned != text:
                        log.info("WS fillers | kind=final utt={} raw={} out={}", job_utt_id, _ws_preview(text), _ws_preview(cleaned))
                        if words and cleaned:
                            aligned = clean_filler_segments([{
                                "index": 1, "start": start, "end": end,
                                "text": text, "punctuation": "", "speaker": None,
                                "chars": [
                                    {"text": w.word, "start": w.start, "end": w.end}
                                    for w in words
                                ],
                            }])
                            chars = aligned[0]["chars"] if aligned else []
                            words = [
                                WordTimestamp(word=c["text"], start=c["start"], end=c["end"])
                                for c in chars
                            ]
                            if words:
                                start, end = words[0].start, words[-1].end
                        else:
                            words = []
                        text = cleaned
                        if not text:
                            log.info("WS drop | reason=filler_only kind=final utt={}", job_utt_id)
                            _release_final_cache()
                            await _send_final_ordered(job_utt_id, None)
                            return
                log.info(
                    "WS final_ready | reason={} utt={} start={:.2f} end={:.2f} "
                    "hard_cut={} chars={} out={}",
                    reason,
                    job_utt_id,
                    start,
                    end,
                    int(hard_cut),
                    len(words),
                    _ws_preview(text),
                )
            else:
                stripped = text
                if job_drop_before is not None or stitch_src:
                    stripped = strip_boundary(
                        stitch_src,
                        text,
                        acoustic_overlap=bool(job_drop_before is not None or audio_start < job_commit_start),
                    )
                    if stripped != text:
                        log.info(
                            "WS partial_strip | utt={} prefix={} raw={} out={} stripped_n={}",
                            job_utt_id,
                            _ws_preview(stitch_src),
                            _ws_preview(text),
                            _ws_preview(stripped),
                            len(text) - len(stripped),
                        )
                # VAD still sees the model's raw punctuation, while provisional
                # display does not inherit a stop at each short PCM boundary.
                raw_partial = stripped
                if tail_partial:
                    committed_raw = partial_committed_vad.get(job_utt_id, "")
                    last_partial_vad_by_utt[job_utt_id] = join_partial_tail(
                        committed_raw, raw_partial
                    )
                else:
                    last_partial_vad_by_utt[job_utt_id] = raw_partial
                text = strip_partial_terminal_punct(raw_partial)
                if tail_partial:
                    prev_shown = partial_committed_text.get(job_utt_id, "")
                    joined = join_partial_tail(prev_shown, text)
                    if joined != text:
                        log.info(
                            "WS partial_tail | utt={} prev={} tail={} out={}",
                            job_utt_id,
                            _ws_preview(prev_shown),
                            _ws_preview(text),
                            _ws_preview(joined),
                        )
                    text = joined
                if ASR_REMOVE_FILLERS:
                    cleaned = clean_fillers(text)
                    if cleaned != text:
                        log.info("WS fillers | kind=partial utt={} raw={} out={}", job_utt_id, _ws_preview(text), _ws_preview(cleaned))
                        text = cleaned
                # Even an empty recognition consumed this exact PCM range.
                # The next partial must start after it, never re-feed it.
                last_partial_audio_end[job_utt_id] = float(audio_start) + (
                    float(audio.size) / TARGET_SR if audio.size else 0.0
                )
                if tail_partial and partial_block_complete:
                    partial_committed_text[job_utt_id] = text
                    partial_committed_vad[job_utt_id] = last_partial_vad_by_utt[job_utt_id]
                    partial_block_start[job_utt_id] = last_partial_audio_end[job_utt_id]
                if not text:
                    log.info(
                        "WS drop | reason=partial_empty utt={} raw={} prefix={}",
                        job_utt_id,
                        _ws_preview(raw_text),
                        _ws_preview(stitch_src),
                    )
                    return
                last_partial_by_utt[job_utt_id] = text
                if len(last_partial_by_utt) > 8:
                    oldest = min(last_partial_by_utt)
                    last_partial_by_utt.pop(oldest, None)
                    last_partial_vad_by_utt.pop(oldest, None)
                    last_partial_audio_end.pop(oldest, None)
                    partial_block_start.pop(oldest, None)
                    partial_committed_text.pop(oldest, None)
                    partial_committed_vad.pop(oldest, None)
                held = pending_caption_prefix(
                    last_partial_by_utt, finalized_utts, job_utt_id
                )
                if held:
                    composed = held + text
                    log.info(
                        "WS partial_hold | utt={} hold={} tail={} out={}",
                        job_utt_id,
                        _ws_preview(held),
                        _ws_preview(text),
                        _ws_preview(composed),
                    )
                    text = composed
                log.info(
                    "WS emit | type=partial reason={} utt={} out={}",
                    reason,
                    job_utt_id,
                    _ws_preview(text),
                )
            prev_emit = last_emitted_by_utt.get(job_utt_id, "")
            text_changed = 1 if text != prev_emit else 0
            last_emitted_by_utt[job_utt_id] = text
            if (
                not is_final
                and text_changed
                and final_submitted_at is not None
                and job_utt_id != final_submit_utt
            ):
                gap = time.perf_counter() - final_submitted_at
                log.info(
                    "WS clocks | kind=partial reason={} utt={} infer_done=1 ws_emit=1 "
                    "text_changed=1 endpoint_wait_ms={:.1f} final_refresh_gap_s={:.3f}",
                    reason,
                    job_utt_id,
                    endpoint_wait_ms,
                    gap,
                )
                final_submitted_at = None
            else:
                log.info(
                    "WS clocks | kind={} reason={} utt={} infer_done=1 ws_emit=1 "
                    "text_changed={} endpoint_wait_ms={:.1f} final_refresh_gap_s=-1",
                    kind,
                    reason,
                    job_utt_id,
                    text_changed,
                    endpoint_wait_ms,
                )
            payload = {
                "text": text,
                "type": "final" if is_final else "partial",
                "utterance_id": job_utt_id,
                "audio_end": round(audio_end, 3),
            }
            if is_final:
                payload["start"] = start
                payload["end"] = end
                payload["speaker"] = None
                if words:
                    payload["chars"] = [
                        {"text": w.word, "start": w.start, "end": w.end} for w in words
                    ]
            if is_final:
                await _send_final_ordered(job_utt_id, payload)
            else:
                await _send(payload)
            if is_final and text and job_utt_id >= carry_from_utt:
                carry_src = raw_for_carry or text
                core = carry_src.strip().rstrip("。！？；!?")
                if len(core) >= 4:
                    carry_prefix = carry_prefix_tail(carry_src, 16)
                    speculative_prefix = carry_prefix
                    carry_from_utt = job_utt_id
                    log.info("WS carry_prefix | utt={} prefix={}", job_utt_id, _ws_preview(carry_prefix))
            _release_final_cache()

        def _spawn(coro) -> asyncio.Task:
            task = asyncio.create_task(coro)
            inflight.add(task)
            task.add_done_callback(inflight.discard)
            return task

        async def _drain() -> None:
            pending = list(inflight)
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        async def emit_asr(
            is_final: bool,
            *,
            hard_cut: bool = False,
            wait: bool = False,
            reason: str = "",
            soft_cut_resume: float | None = None,
            resumed_after_silence_ms: float | None = None,
            hard_cut_plan=None,
        ) -> None:
            nonlocal speech_buf, last_partial_at, carry_prefix, speculative_prefix
            nonlocal lazy_partial_pending, final_submitted_at, final_submit_utt
            nonlocal endpoint_wait_started
            if not reason:
                reason = "hard_cut" if hard_cut else ("final" if is_final else "partial")
            if not is_final and partial_gate.defer_if_running(utt_id):
                return
            if not is_final and WS_LAZY_PARTIAL and infer_queue.busy:
                lazy_partial_pending = True
                return
            if not is_final:
                lazy_partial_pending = False
            if speech_buf.size == 0:
                if is_final:
                    t = round(_stream_cursor(), 2)
                    log.info("WS emit | type=final reason={} empty_buf cursor={:.2f}", reason, t)
                    await _send({
                        "text": "",
                        "type": "final",
                        "start": t,
                        "end": t,
                        "speaker": None,
                    })
                return

            job_utt_id = utt_id
            prefix = speculative_prefix or carry_prefix
            audio_start = buffer_audio_start
            job_commit = commit_start
            job_drop = drop_before
            audio = speech_buf.copy()
            tail_partial = False
            window = None
            wait_ms = 0.0
            if endpoint_wait_started:
                wait_ms = max(0.0, (time.perf_counter() - endpoint_wait_started) * 1000.0)
                endpoint_wait_started = 0.0
            if is_final:
                final_submitted_at = time.perf_counter()
                final_submit_utt = job_utt_id
                lazy_partial_pending = False
                if hard_cut:
                    guess = last_partial_by_utt.get(job_utt_id, "")
                    if guess:
                        speculative_prefix = carry_prefix_tail(stitch_prev_text(guess), 16)
                    cut_audio = _apply_hard_cut(hard_cut_plan)
                    if cut_audio is not None:
                        audio = cut_audio
                elif reason == "soft_cut" and soft_cut_resume is not None:
                    guess = last_partial_by_utt.get(job_utt_id, "")
                    if guess:
                        speculative_prefix = carry_prefix_tail(stitch_prev_text(guess), 16)
                    buf_end = audio_start + (speech_buf.size / TARGET_SR if speech_buf.size else 0.0)
                    plan = plan_soft_cut(
                        audio_start,
                        soft_cut_resume,
                        float(resumed_after_silence_ms or 0.0),
                        WS_SOFT_CUT_LOOKBACK_MS,
                        buf_end,
                    )
                    audio = _apply_soft_cut(plan)
                elif reason == "vad_end":
                    guess = last_partial_by_utt.get(job_utt_id, "")
                    if guess:
                        speculative_prefix = carry_prefix_tail(stitch_prev_text(guess), 16)
                    _apply_vad_turn()
                else:
                    _begin_next_utterance(reason, reset_vad=True, hangover=False)
                    carry_prefix = ""
                    speculative_prefix = ""
                    carry_from_utt = -1
            origin_audio = audio.copy() if WS_SHARED_UTT else None
            origin_start = audio_start
            if audio.size == 0:
                if is_final:
                    t = round(_stream_cursor(), 2)
                    log.info("WS emit | type=final reason={} empty_split cursor={:.2f}", reason, t)
                    await _send_final_ordered(job_utt_id, {
                        "text": "",
                        "type": "final",
                        "utterance_id": job_utt_id,
                        "audio_end": t,
                        "start": t,
                        "end": t,
                        "speaker": None,
                    })
                return
            if not is_final and WS_PARTIAL_WINDOW_SEC > 0:
                cap = int(round(WS_PARTIAL_WINDOW_SEC * TARGET_SR))
                window = plan_partial_window(
                    int(audio.size),
                    cap,
                    has_shown=job_utt_id in last_partial_audio_end,
                    sample_rate=TARGET_SR,
                    origin_start=audio_start,
                    prev_window_end=last_partial_audio_end.get(job_utt_id),
                    block_start=partial_block_start.get(job_utt_id),
                )
                if window is not None:
                    if window.keep <= 0:
                        return
                    audio = audio[window.offset:window.offset + window.keep].copy()
                    audio_start = audio_start + window.offset / TARGET_SR
                    tail_partial = window.tail
                    suffix = "_head" if window.head else ("_step" if window.step else "")
                    log.info(
                        "WS partial_window{} | utt={} keep={:.3f}s audio={:.3f}-{:.3f} "
                        "block_complete={}",
                        suffix,
                        job_utt_id,
                        window.keep / TARGET_SR,
                        audio_start,
                        audio_start + window.keep / TARGET_SR,
                        int(window.complete),
                    )
            if client_sr == 8000:
                audio = _telephony_bandpass(audio, TARGET_SR)
            log.info(
                "WS submit | reason={} kind={} utt={} audio={:.3f}-{:.3f} "
                "buf={:.3f}s commit={:.3f} drop_before={} prefix={} wait={}",
                reason,
                "final" if is_final else "partial",
                job_utt_id,
                audio_start,
                audio_start + (audio.size / TARGET_SR),
                audio.size / TARGET_SR,
                job_commit,
                job_drop,
                _ws_preview(prefix),
                int(wait),
            )
            task = _spawn(_run_and_send(
                job_utt_id,
                is_final,
                audio,
                audio_start,
                job_commit,
                job_drop,
                prefix,
                hard_cut,
                reason,
                tail_partial,
                bool(window and window.complete),
                origin_audio,
                origin_start,
                wait_ms,
            ))
            if not is_final:
                partial_gate.started(job_utt_id, more_audio=bool(window and window.step))

                def _partial_finished(_task: asyncio.Task) -> None:
                    nonlocal last_partial_at
                    if (partial_gate.finished(job_utt_id, utt_id)
                            and accept_pcm and speech_buf.size > 0):
                        # The next received packet submits one fresh range after
                        # this result has advanced last_partial_audio_end. Avoid
                        # racing another queued snapshot against that update.
                        last_partial_at = min(
                            last_partial_at,
                            max(0, speech_buf.size - partial_samples),
                        )

                task.add_done_callback(_partial_finished)
            if wait:
                await task

        def _flush_lazy_partial() -> None:
            if not lazy_partial_pending or infer_queue.busy:
                return
            asyncio.create_task(emit_asr(False, reason="partial"))

        _flush_lazy_cb = _flush_lazy_partial
        infer_queue.add_idle_callback(_flush_lazy_cb)

        while True:
            data = await websocket.receive()
            if data.get("type") == "websocket.disconnect":
                break

            if "text" in data:
                try:
                    msg = json.loads(data["text"])
                except json.JSONDecodeError:
                    await _send({
                        "code": "INVALID_JSON",
                        "message": "Invalid JSON command",
                        "statusCode": 400,
                    })
                    continue
                action = msg.get("action", "")
                if action == "flush":
                    await emit_asr(True, wait=True, reason="flush")
                elif action == "reset":
                    partial_gate.reset()
                    await _drain()
                    log.info(
                        "WS reset | utt={} cursor={:.3f} buf={:.3f}s",
                        utt_id,
                        _stream_cursor(),
                        speech_buf.size / TARGET_SR if speech_buf.size else 0.0,
                    )
                    speech_buf = np.zeros(0, dtype=np.float32)
                    pre_roll = np.zeros(0, dtype=np.float32)
                    last_partial_at = 0
                    last_partial_by_utt.clear()
                    last_partial_vad_by_utt.clear()
                    last_partial_audio_end.clear()
                    partial_block_start.clear()
                    partial_committed_text.clear()
                    partial_committed_vad.clear()
                    last_emitted_by_utt.clear()
                    finalized_utts.clear()
                    hangover_left = 0
                    hangover_heard = False
                    carry_prefix = ""
                    speculative_prefix = ""
                    carry_from_utt = -1
                    punct_defer = False
                    punct_defer_silence = 0
                    energy_run = 0
                    energy_hold = False
                    energy_silence = 0
                    turn_overlap = np.zeros(0, dtype=np.float32)
                    old_utt = utt_id
                    utt_id += 1
                    final_buffer.reset(utt_id)
                    backend.drop_utt_cache(ws_req_id, old_utt)
                    vad.reset()
                    buffer_audio_start = _stream_cursor()
                    commit_start = buffer_audio_start
                    drop_before = None
                    await _send({"status": "buffer_reset"})
                elif action == "config":
                    new_lang = msg.get("language")
                    if new_lang == "auto":
                        lang_code = None
                    elif new_lang:
                        lang_code = new_lang
                    if "use_server_vad" in msg:
                        use_vad = bool(msg["use_server_vad"])
                    await _send({
                        "status": "configured",
                        "language": lang_code or "auto",
                        "use_server_vad": use_vad,
                    })
                elif action == "stop":
                    accept_pcm = False
                    if speech_buf.size > 0:
                        await emit_asr(True, wait=True, reason="stop")
                    await _drain()
                    await infer_queue.wait_idle()
                    log.info("WS session stopped, model kept loaded")
                    await _send({"status": "stopped", "model_loaded": backend.loaded})
                    break
                else:
                    await _send({
                        "code": "UNKNOWN_ACTION",
                        "message": f"Unknown action: {action!r}",
                        "statusCode": 400,
                    })
                continue

            if "bytes" not in data or not accept_pcm:
                continue

            incoming = data["bytes"]
            if client_sr != TARGET_SR:
                incoming = _resample_pcm_bytes(incoming, client_sr)
            chunk = _pcm_to_float(incoming)
            pre_roll = append_pre_roll(pre_roll, chunk, pre_roll_samples)
            event = vad.accept_audio(chunk) if use_vad else type("E", (), {"is_speech": True, "speech_ended": False})()
            stream_samples += int(chunk.size)
            endpoint_resumed = False

            if punct_defer and event.is_speech:
                log.info(
                    "WS vad_defer_resume | utt={} cursor={:.3f} waited={:.3f}s",
                    utt_id,
                    _stream_cursor(),
                    punct_defer_silence / TARGET_SR,
                )
                endpoint_resumed = True
                punct_defer = False
                punct_defer_silence = 0

            if punct_defer and not event.is_speech:
                punct_defer_silence += int(chunk.size)
                # 等待期间的静音 chunk 必须写入 speech_buf，
                # 否则 emit_asr 提交的音频缺少这段静音，
                # 说话人在等待期间说的话（如"能没这个"）会被丢弃。
                speech_buf = np.concatenate([speech_buf, chunk]) if speech_buf.size else chunk.copy()
                guess = last_partial_vad_by_utt.get(utt_id, "")
                if punct_defer_silence >= punct_defer_cap:
                    log.info(
                        "WS vad_end | reason=resume_grace_timeout utt={} cursor={:.3f} "
                        "buf={:.3f}s extra_silence={:.3f}s sentence_end={} partial={}",
                        utt_id,
                        _stream_cursor(),
                        speech_buf.size / TARGET_SR if speech_buf.size else 0.0,
                        punct_defer_silence / TARGET_SR,
                        int(text_ends_sentence(guess)),
                        _ws_preview(guess),
                    )
                    punct_defer = False
                    punct_defer_silence = 0
                    await emit_asr(True, reason="vad_end")
                continue

            prev_energy_run = energy_run
            energy_run = advance_energy_run(energy_run, chunk)
            loud = energy_run > prev_energy_run
            energy_open = (
                use_vad
                and not event.is_speech
                and speech_buf.size == 0
                and hangover_left <= 0
                and energy_onset_ready(energy_run, TARGET_SR)
            )
            if energy_open:
                energy_hold = True
                energy_silence = 0
                log.info(
                    "WS energy_onset | utt={} cursor={:.3f} run={:.3f}s threshold={:.3f}",
                    utt_id,
                    _stream_cursor(),
                    energy_run / TARGET_SR,
                    ENERGY_ONSET_RMS,
                )
            keep_audio = (
                bool(event.is_speech)
                or speech_buf.size > 0
                or hangover_left > 0
                or energy_open
            )
            if keep_audio:
                after_skip = discard_stale_overlap_after_skip(skip_samples)
                if skip_samples > 0:
                    log.warning(
                        "WS audio_resume | skipped={:.3f}s from={:.3f} to={:.3f} utt={}",
                        skip_samples / TARGET_SR,
                        skip_from,
                        _stream_cursor(),
                        utt_id,
                    )
                    skip_samples = 0
                    last_skip_log = 0
                    if after_skip and turn_overlap.size:
                        log.info(
                            "WS onset_fresh | utt={} discarded_stale_overlap={:.3f}s",
                            utt_id,
                            turn_overlap.size / TARGET_SR,
                        )
                        turn_overlap = np.zeros(0, dtype=np.float32)
                    if after_skip:
                        carry_prefix = ""
                        speculative_prefix = ""
                        carry_from_utt = -1
                if speech_buf.size == 0:
                    if hangover_left > 0 and not event.is_speech:
                        pass
                    elif turn_overlap.size > 0:
                        overlap_n = int(turn_overlap.size)
                        speech_buf = turn_overlap
                        turn_overlap = np.zeros(0, dtype=np.float32)
                        buffer_audio_start, boundary = retie_turn_overlap(
                            overlap_n,
                            int(chunk.size),
                            stream_samples,
                            TARGET_SR,
                        )
                        commit_start = boundary
                        drop_before = boundary
                        log.info(
                            "WS onset | utt={} cursor={:.3f} reuse_overlap={:.3f}s "
                            "pre_roll=0.000s packet={:.3f}s audio_start={:.3f} "
                            "drop_before={:.3f} hangover={}",
                            utt_id,
                            _stream_cursor(),
                            overlap_n / TARGET_SR,
                            chunk.size / TARGET_SR,
                            buffer_audio_start,
                            drop_before,
                            int(hangover_left > 0),
                        )
                    else:
                        unused = unused_pre_roll(pre_roll, chunk)
                        if unused.size:
                            speech_buf = unused
                        onset_n = int(speech_buf.size) + int(chunk.size)
                        buffer_audio_start = (stream_samples - onset_n) / TARGET_SR
                        commit_start = buffer_audio_start
                        drop_before = None
                        log.info(
                            "WS onset | utt={} cursor={:.3f} pre_roll={:.3f}s packet={:.3f}s "
                            "audio_start={:.3f} vad_speech={} energy={} hangover={}",
                            utt_id,
                            _stream_cursor(),
                            unused.size / TARGET_SR,
                            chunk.size / TARGET_SR,
                            buffer_audio_start,
                            int(bool(event.is_speech)),
                            int(energy_open),
                            int(hangover_left > 0),
                        )
                speech_buf = np.concatenate([speech_buf, chunk]) if speech_buf.size else chunk.copy()
                if hangover_left > 0:
                    if event.is_speech:
                        hangover_heard = True
                        hangover_left = 0
                        if turn_overlap.size:
                            overlap_n = int(turn_overlap.size)
                            speech_buf = np.concatenate([turn_overlap, speech_buf])
                            turn_overlap = np.zeros(0, dtype=np.float32)
                            buffer_audio_start, boundary = retie_turn_overlap(
                                overlap_n,
                                int(speech_buf.size) - overlap_n,
                                stream_samples,
                                TARGET_SR,
                            )
                            commit_start = boundary
                            drop_before = boundary
                        log.info(
                            "WS hangover_promote | utt={} cursor={:.3f} buf={:.3f}s "
                            "audio_start={:.3f} drop_before={}",
                            utt_id,
                            _stream_cursor(),
                            speech_buf.size / TARGET_SR,
                            buffer_audio_start,
                            drop_before,
                        )
                    else:
                        hangover_left -= int(chunk.size)
                        if hangover_left <= 0 and not hangover_heard:
                            hangover_s = speech_buf.size / TARGET_SR if speech_buf.size else 0.0
                            if speech_buf.size:
                                turn_overlap = (
                                    np.concatenate([turn_overlap, speech_buf])
                                    if turn_overlap.size
                                    else speech_buf.copy()
                                )
                            log.info(
                                "WS hangover_discard | utt={} cursor={:.3f} kept_hangover={:.3f}s "
                                "keep_overlap={:.3f}s",
                                utt_id,
                                _stream_cursor(),
                                hangover_s,
                                turn_overlap.size / TARGET_SR,
                            )
                            speech_buf = np.zeros(0, dtype=np.float32)
                            last_partial_at = 0
                            hangover_left = 0
            else:
                if skip_samples == 0:
                    skip_from = _stream_cursor() - (chunk.size / TARGET_SR)
                    log.info(
                        "WS audio_skip_start | cursor={:.3f} utt={} vad_speech=0 buf=0",
                        skip_from,
                        utt_id,
                    )
                skip_samples += int(chunk.size)
                if skip_samples - last_skip_log >= int(2 * TARGET_SR):
                    log.warning(
                        "WS audio_skipping | dropped={:.3f}s from={:.3f} cursor={:.3f} utt={}",
                        skip_samples / TARGET_SR,
                        skip_from,
                        _stream_cursor(),
                        utt_id,
                    )
                    last_skip_log = skip_samples

            force_speech_ended = False
            if energy_hold and keep_audio and not event.speech_ended:
                if event.is_speech:
                    energy_hold = False
                    energy_silence = 0
                elif loud:
                    energy_silence = 0
                else:
                    energy_silence += int(chunk.size)
                    speech_n = max(0, int(speech_buf.size) - energy_silence)
                    needed = silence_threshold_samples(
                        speech_n,
                        TARGET_SR,
                        WS_FLUSH_SILENCE_MS,
                        WS_EARLY_SILENCE_MS,
                        WS_COMMITTED_SPEECH_SECONDS,
                    )
                    if speech_n > 0 and energy_silence >= needed:
                        log.info(
                            "WS energy_end | utt={} cursor={:.3f} silence={:.3f}s buf={:.3f}s",
                            utt_id,
                            _stream_cursor(),
                            energy_silence / TARGET_SR,
                            speech_buf.size / TARGET_SR if speech_buf.size else 0.0,
                        )
                        force_speech_ended = True
                        energy_hold = False
                        energy_silence = 0
                        energy_run = 0

            fresh_samples = fresh_audio_samples(
                speech_buf.size,
                buffer_audio_start,
                commit_start,
                TARGET_SR,
            )
            if fresh_samples >= max_samples:
                audio_duration = speech_buf.size / TARGET_SR
                reused_context = max(0.0, commit_start - buffer_audio_start)
                target_sec = min(
                    audio_duration,
                    reused_context + WS_MAX_UTTERANCE_SECONDS,
                )
                silences = await asyncio.to_thread(
                    scan_silences,
                    speech_buf.copy(),
                    TARGET_SR,
                )
                silence = choose_hard_cut_silence(
                    silences,
                    target_sec,
                    WS_HARD_CUT_SEARCH_SECONDS,
                    WS_HARD_CUT_MIN_SILENCE_MS,
                )
                hard_cut_plan = None
                if silence is not None:
                    silence_start, silence_end = silence
                    hard_cut_plan = plan_soft_cut(
                        buffer_audio_start,
                        buffer_audio_start + silence_end,
                        (silence_end - silence_start) * 1000.0,
                        WS_SOFT_CUT_LOOKBACK_MS,
                        buffer_audio_start + audio_duration,
                    )
                    log.info(
                        "WS hard_cut_silence | target={:.3f}s region={:.3f}-{:.3f}s "
                        "boundary={:.3f}s",
                        target_sec,
                        silence_start,
                        silence_end,
                        hard_cut_plan.previous_end - buffer_audio_start,
                    )
                else:
                    log.info(
                        "WS hard_cut_silence | no_candidate target={:.3f}s search={:.3f}s "
                        "min_silence_ms={}",
                        target_sec,
                        WS_HARD_CUT_SEARCH_SECONDS,
                        WS_HARD_CUT_MIN_SILENCE_MS,
                    )
                await emit_asr(
                    True,
                    hard_cut=True,
                    reason="hard_cut",
                    hard_cut_plan=hard_cut_plan,
                )
                continue

            if use_vad and (event.speech_ended or force_speech_ended):
                guess = last_partial_vad_by_utt.get(utt_id, "")
                punct_defer = True
                punct_defer_silence = 0
                endpoint_wait_started = time.perf_counter()
                log.info(
                    "WS vad_defer | utt={} cursor={:.3f} buf={:.3f}s "
                    "cap={:.3f}s sentence_end={} partial={}",
                    utt_id,
                    _stream_cursor(),
                    speech_buf.size / TARGET_SR if speech_buf.size else 0.0,
                    WS_VAD_END_MAX_SILENCE_MS / 1000.0,
                    int(text_ends_sentence(guess)),
                    _ws_preview(guess),
                )
                continue

            # A resumed pause can soft-cut after the minimum utterance duration when its
            # silence is within the soft-cut/flush range; punctuation is logged but does not gate it.
            if (not endpoint_resumed
                    and WS_SOFT_CUT_ENABLED
                    and event.speech_resumed
                    and WS_SOFT_CUT_SILENCE_MS <= event.resumed_after_silence_ms < WS_FLUSH_SILENCE_MS):
                resume_t = _stream_cursor() - (chunk.size / TARGET_SR)
                resume_t = max(buffer_audio_start, min(resume_t, _stream_cursor()))
                utterance_duration = resume_t - buffer_audio_start
                partial_guess = last_partial_by_utt.get(utt_id, "")
                sentence_ready = text_ends_sentence(partial_guess)
                if utterance_duration >= WS_SOFT_CUT_START_SECONDS:
                    log.info(
                        "WS soft_cut | utt={} cursor={:.3f} resume={:.3f} "
                        "utterance_dur={:.3f}s resumed_silence={:.0f}ms sentence_end={} buf={:.3f}s",
                        utt_id,
                        _stream_cursor(),
                        resume_t,
                        utterance_duration,
                        event.resumed_after_silence_ms,
                        int(sentence_ready),
                        speech_buf.size / TARGET_SR if speech_buf.size else 0.0,
                    )
                    await emit_asr(
                        True,
                        reason="soft_cut",
                        soft_cut_resume=resume_t,
                        resumed_after_silence_ms=event.resumed_after_silence_ms,
                    )
                    continue

            interval_samples = partial_samples
            if speech_buf.size >= last_partial_at + interval_samples and speech_buf.size >= partial_samples:
                last_partial_at = speech_buf.size
                await emit_asr(False, reason="partial")

    except WebSocketDisconnect:
        log.info("[WS] Client disconnected")
    except Exception as e:
        log.error("WebSocket error: {}", e)
        try:
            await websocket.send_json({"code": "WEBSOCKET_ERROR", "message": str(e), "statusCode": 500})
        except Exception:
            pass
    finally:
        try:
            if speech_buf.size > 0:
                await emit_asr(True, wait=True, reason="disconnect")
            await _drain()
        except Exception:
            pass
        if _flush_lazy_cb is not None:
            infer_queue.remove_idle_callback(_flush_lazy_cb)
        backend.drop_session_cache(ws_req_id)
        reset_request_id(token)
        async with _ws_lock:
            if _active_ws is websocket:
                _active_ws = None
        _touch()
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
