from __future__ import annotations

import gc
import os
import sys
import threading
import numpy as np
from scipy.signal import resample_poly

from backend.base import ASRBackend, ASRResult, BackendState, WordTimestamp
from config import (
    ALIGNER_BACKEND_FN,
    ALIGNER_FRONTEND_FN,
    ALIGNER_LLM_FN,
    DML_PAD_TO,
    ENABLE_ALIGNER,
    ENCODER_BACKEND_FN,
    ENCODER_FRONTEND_FN,
    FILE_CHUNK_SIZE_SEC,
    GGML_VK_DISABLE_F16,
    LLM_FN,
    LLM_USE_GPU,
    MODEL_DIR,
    ONNX_PROVIDER,
    REALTIME_CHUNK_SIZE_SEC,
    ASR_TEMPERATURE,
    WS_ASR_TEMPERATURE,
    OFFLINE_ASR_TEMPERATURE,
    WS_FINAL_ALIGN,
    WS_FINAL_YIELD,
    WS_FIXED_ENCODER_WINDOW,
    WS_SHARED_UTT,
    TARGET_SR,
    VENDOR_DIR,
    apply_llm_device_env,
)
from logger import log
from utterance_cache import UtteranceEncoderCache, full_window_count
from ws_overlap import realtime_window_sec

_ABORT_MARK = "====解码有误，强制熔断===="

_LANG_MAP = {
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-hans": "Chinese",
    "cn": "Chinese",
    "en": "English",
    "yue": "Cantonese",
    "zh-hk": "Cantonese",
    "zh-yue": "Cantonese",
    "ja": "Japanese",
    "ko": "Korean",
    "th": "Thai",
    "hi": "Hindi",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "ru": "Russian",
    "vi": "Vietnamese",
    "tr": "Turkish",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "el": "Greek",
    "ro": "Romanian",
    "hu": "Hungarian",
}


def _ensure_vendor_path() -> None:
    vendor = str(VENDOR_DIR)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)


def map_language(language: str | None) -> str | None:
    if language is None:
        return None
    raw = str(language).strip()
    if not raw or raw.lower() == "auto":
        return None
    mapped = _LANG_MAP.get(raw.lower())
    if mapped:
        return mapped
    return raw[:1].upper() + raw[1:].lower() if raw else None


def public_language(internal: str | None, requested: str | None) -> str:
    if requested and requested.lower() not in ("auto", ""):
        return requested.lower() if len(requested) <= 8 else requested
    if internal == "Chinese":
        return "zh"
    if internal == "English":
        return "en"
    if internal:
        return internal
    return "auto"


def _to_16k_mono(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    wav = np.asarray(audio, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1).astype(np.float32)
    if sample_rate != TARGET_SR:
        from math import gcd
        g = gcd(int(sample_rate), TARGET_SR)
        wav = resample_poly(wav, TARGET_SR // g, int(sample_rate) // g).astype(np.float32)
    return wav


def _encoder_fingerprint() -> str:
    return f"{ENCODER_FRONTEND_FN}|{ENCODER_BACKEND_FN}|{REALTIME_CHUNK_SIZE_SEC}|{DML_PAD_TO}"


def _perf_view(perf: dict) -> dict:
    return {
        "encode_ms": round(float(perf.get("encode_time", 0) or 0) * 1000.0, 1),
        "prefill_ms": round(float(perf.get("prefill_time", 0) or 0) * 1000.0, 1),
        "decode_ms": round(float(perf.get("decode_time", 0) or 0) * 1000.0, 1),
        "n_chunks": int(perf.get("n_chunks") or 0),
        "cache_hits": int(perf.get("cache_hits") or 0),
        "decode_skipped": int(perf.get("decode_skipped") or 0),
        "committed_chunks": int(perf.get("committed_chunks") or 0),
        "yielded": int(perf.get("yielded") or 0),
        "next_chunk": int(perf.get("next_chunk") or 0),
    }


def _strip_abort(text: str) -> str:
    if not text:
        return ""
    return text.replace(_ABORT_MARK, "").strip()


class QwenGGUFBackend(ASRBackend):
    def __init__(self):
        self._lock = threading.Lock()
        self._state = BackendState.UNLOADED
        self._engine = None
        self._utt_cache = UtteranceEncoderCache(fingerprint=_encoder_fingerprint())

    @property
    def state(self) -> BackendState:
        return self._state

    def load(self) -> None:
        with self._lock:
            if self._state == BackendState.READY and self._engine is not None:
                return
            if self._state in (BackendState.LOADING, BackendState.UNLOADING):
                raise RuntimeError(f"Cannot load while backend is {self._state.value}")
            self._state = BackendState.LOADING
        try:
            apply_llm_device_env()
            _ensure_vendor_path()
            from qwen_asr_gguf.inference import ASREngineConfig, QwenASREngine
            from qwen_asr_gguf.inference.schema import AlignerConfig

            align_config = None
            if ENABLE_ALIGNER:
                align_config = AlignerConfig(
                    model_dir=str(MODEL_DIR),
                    encoder_frontend_fn=ALIGNER_FRONTEND_FN,
                    encoder_backend_fn=ALIGNER_BACKEND_FN,
                    llm_fn=ALIGNER_LLM_FN,
                    onnx_provider=ONNX_PROVIDER,
                    llm_use_gpu=LLM_USE_GPU,
                    dml_pad_to=max(DML_PAD_TO, int(FILE_CHUNK_SIZE_SEC)),
                )
            config = ASREngineConfig(
                model_dir=str(MODEL_DIR),
                encoder_frontend_fn=ENCODER_FRONTEND_FN,
                encoder_backend_fn=ENCODER_BACKEND_FN,
                llm_fn=LLM_FN,
                onnx_provider=ONNX_PROVIDER,
                llm_use_gpu=LLM_USE_GPU,
                enable_aligner=ENABLE_ALIGNER,
                align_config=align_config,
                verbose=False,
                dml_pad_to=DML_PAD_TO,
            )
            log.info(
                "Loading GGUF engine | dir={} llm={} provider={} vulkan={} "
                "llm_gpu={} vk_disable_f16={} aligner={}",
                MODEL_DIR,
                LLM_FN,
                ONNX_PROVIDER,
                os.environ.get("GGML_VULKAN"),
                LLM_USE_GPU,
                GGML_VK_DISABLE_F16,
                ENABLE_ALIGNER,
            )
            engine = QwenASREngine(config)
            with self._lock:
                self._engine = engine
                self._state = BackendState.READY
            log.info("GGUF engine ready")
        except Exception:
            with self._lock:
                self._engine = None
                self._state = BackendState.UNLOADED
            raise

    def _require_ready(self):
        if self._state != BackendState.READY or self._engine is None:
            raise RuntimeError(f"Backend not ready ({self._state.value})")

    def _run(
        self,
        audio,
        sample_rate: int,
        language: str | None,
        chunk_size_sec: float,
        is_final: bool,
        do_align: bool,
        on_chunk=None,
        prefix_text: str = "",
        temperature: float = ASR_TEMPERATURE,
        abort_event: "threading.Event | None" = None,
        encoder_cache=None,
        skip_decode_before: int = 0,
        yield_event: "threading.Event | None" = None,
    ) -> ASRResult:
        with self._lock:
            self._require_ready()
            engine = self._engine
        wav = _to_16k_mono(audio, sample_rate)
        internal_lang = map_language(language)
        try:
            result = engine.asr(
                audio=wav,
                context="",
                language=internal_lang,
                chunk_size_sec=chunk_size_sec,
                temperature=temperature,
                streaming=False,
                do_align=do_align and ENABLE_ALIGNER,
                on_chunk=on_chunk,
                prefix_text=prefix_text,
                abort_event=abort_event,
                encoder_cache=encoder_cache,
                skip_decode_before=skip_decode_before,
                yield_event=yield_event,
            )
        except Exception:
            log.exception("ASR decode failed; reload after ASR failure")
            try:
                self.unload()
                self.load()
            except Exception:
                log.exception("Engine reload after ASR failure also failed")
            raise
        text = _strip_abort(getattr(result, "text", "") or "")
        words = None
        alignment = getattr(result, "alignment", None)
        if alignment and getattr(alignment, "items", None):
            words = [
                WordTimestamp(
                    word=item.text,
                    start=round(float(item.start_time), 3),
                    end=round(float(item.end_time), 3),
                )
                for item in alignment.items
            ]
        perf = getattr(result, "performance", None) or {}
        n_gen = int(perf.get("decode_tokens", 0) or 0)
        return ASRResult(
            text=text,
            language=public_language(internal_lang, language),
            is_final=is_final,
            words=words,
            n_generate=n_gen,
            perf=_perf_view(perf),
        )

    def transcribe_file(
        self,
        audio,
        sample_rate: int,
        language: str | None = None,
        on_chunk=None,
        prefix_text: str = "",
        chunk_size_sec: float | None = None,
    ) -> ASRResult:
        size = FILE_CHUNK_SIZE_SEC if chunk_size_sec is None else float(chunk_size_sec)
        return self._run(
            audio,
            sample_rate,
            language,
            size,
            is_final=True,
            do_align=True,
            on_chunk=on_chunk,
            prefix_text=prefix_text,
            temperature=OFFLINE_ASR_TEMPERATURE,
        )

    def advance_encoder(
        self,
        audio,
        sample_rate: int,
        session_id: str | None,
        utt_id: int | None,
        audio_start_sample: int,
    ) -> dict:
        """Encode missing full windows from the utterance origin. No decode."""
        if not WS_SHARED_UTT:
            return {"encoded": 0, "n_full": 0}
        wav = _to_16k_mono(audio, sample_rate)
        sr = int(sample_rate or TARGET_SR)
        window_samples = max(1, int(round(float(REALTIME_CHUNK_SIZE_SEC) * sr)))
        n_full = full_window_count(int(wav.size), window_samples)
        chunks = self._utt_cache.ensure_origin(session_id, utt_id, audio_start_sample)
        if chunks is None:
            return {"encoded": 0, "n_full": n_full}
        encoded = 0
        with self._lock:
            self._require_ready()
            engine = self._engine
            for idx in range(n_full):
                from qwen_asr_gguf.inference.chunk_cache import cache_embd, cache_entry, set_cache_embd

                if cache_embd(cache_entry(chunks, idx)) is not None:
                    continue
                start = idx * window_samples
                piece = wav[start:start + window_samples]
                if piece.size < window_samples:
                    continue
                audio_feature, _enc_time = engine.encoder.encode(piece)
                if audio_feature is not None:
                    set_cache_embd(chunks, idx, np.copy(audio_feature))
                    encoded += 1
        if encoded:
            log.info(
                "WS advance_encoder | utt={} origin={} n_full={} encoded={}",
                utt_id,
                audio_start_sample,
                n_full,
                encoded,
            )
        return {"encoded": encoded, "n_full": n_full}

    def transcribe_realtime(
        self,
        audio,
        sample_rate: int,
        language: str | None = None,
        is_final: bool = False,
        prefix_text: str = "",
        abort_event: "threading.Event | None" = None,
        session_id: str | None = None,
        utt_id: int | None = None,
        audio_start_sample: int = 0,
        use_utt_cache: bool = True,
        origin_audio=None,
        origin_start_sample: int | None = None,
        yield_event: "threading.Event | None" = None,
    ) -> ASRResult:
        wav = np.asarray(audio).reshape(-1)
        sr = int(sample_rate or TARGET_SR)
        duration = float(wav.size) / float(sr) if wav.size and sr else 0.0
        origin_start = audio_start_sample if origin_start_sample is None else int(origin_start_sample)
        if WS_SHARED_UTT and origin_audio is not None:
            self.advance_encoder(origin_audio, sample_rate, session_id, utt_id, origin_start)
        if WS_FIXED_ENCODER_WINDOW:
            window = float(REALTIME_CHUNK_SIZE_SEC)
        else:
            window = realtime_window_sec(duration, REALTIME_CHUNK_SIZE_SEC)
        encoder_cache = None
        if use_utt_cache and WS_SHARED_UTT:
            encoder_cache = self._utt_cache.ensure_origin(session_id, utt_id, origin_start)

        skip_before = 0
        if WS_FINAL_YIELD:
            resume = self._utt_cache.get_resume(session_id, utt_id)
            if resume:
                skip_before = int(resume.get("next_idx") or 0)

        def _once(chunk_sec: float, prefix: str) -> ASRResult:
            return self._run(
                audio,
                sample_rate,
                language,
                chunk_sec,
                is_final=is_final,
                do_align=is_final and WS_FINAL_ALIGN,
                prefix_text=prefix,
                temperature=WS_ASR_TEMPERATURE,
                abort_event=abort_event,
                encoder_cache=encoder_cache,
                skip_decode_before=skip_before,
                yield_event=yield_event if (WS_FINAL_YIELD and is_final) else None,
            )

        # Decoder text prefix after a finished sentence (e.g. "正确的方向。")
        # frequently yields immediate EOS. Continuity is the 1s PCM overlap.
        result = _once(window, "")
        if result.perf and result.perf.get("yielded"):
            self._utt_cache.set_resume(session_id, utt_id, {
                "next_idx": int(result.perf.get("next_chunk") or 0),
            })
        else:
            self._utt_cache.set_resume(session_id, utt_id, None)
        if (result.text or "").strip():
            return result
        # 空结果不再同参数 retry（温度 0 重跑必空，纯耗时）。
        # 上层 _run_and_send 的 fallback_empty_final 会用最近 partial 兜底。
        if duration > REALTIME_CHUNK_SIZE_SEC + 0.2:
            log.warning(
                "WS_EMPTY_RESULT | window={:.2f}s duration={:.2f}s dropped_prefix={}",
                window,
                duration,
                bool(prefix_text),
            )
        elif prefix_text:
            log.warning(
                "WS_EMPTY_RESULT | window={:.2f}s duration={:.2f}s (prefix ignored)",
                window,
                duration,
            )
        return result

    def drop_utt_cache(self, session_id: str | None, utt_id: int | None) -> None:
        self._utt_cache.drop_utt(session_id, utt_id)

    def drop_session_cache(self, session_id: str | None) -> None:
        self._utt_cache.drop_session(session_id)

    def unload(self) -> None:
        with self._lock:
            if self._state == BackendState.UNLOADED and self._engine is None:
                return
            if self._state == BackendState.LOADING:
                raise RuntimeError("Cannot unload while LOADING")
            self._state = BackendState.UNLOADING
            engine = self._engine
            self._engine = None
            self._utt_cache = UtteranceEncoderCache(fingerprint=_encoder_fingerprint())
        try:
            if engine is not None:
                engine.close()
        finally:
            gc.collect()
            with self._lock:
                self._state = BackendState.UNLOADED
            log.info("GGUF engine unloaded")
