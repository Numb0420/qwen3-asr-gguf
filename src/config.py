"""Runtime configuration for the local GGUF ASR service."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from logger import log
from paths import ROOT_DIR, SRC_DIR, VENDOR_DIR, default_log_dir


def _safe_float(name: str, default: str) -> float:
    raw = os.getenv(name, default)
    try:
        return float(raw)
    except ValueError:
        log.error("Config error: {} must be a float, got '{}' — using default {}", name, raw, default)
        return float(default)


def _safe_int(name: str, default: str) -> int:
    raw = os.getenv(name, default)
    try:
        return int(raw)
    except ValueError:
        log.error("Config error: {} must be an integer, got '{}' — using default {}", name, raw, default)
        return int(default)


def _safe_bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


_VALID_LOG_LEVELS = {"TRACE", "DEBUG", "INFO", "WARNING", "WARN", "ERROR", "CRITICAL", "FATAL"}
_LOG_LEVEL_ALIASES = {"WARN": "WARNING", "FATAL": "CRITICAL"}
_VALID_ONNX_PROVIDERS = {"DML", "CPU", "CUDA", "TRT", "TENSORRT"}

MODEL_DIR = Path(os.getenv("MODEL_DIR", str(ROOT_DIR / "models"))).expanduser()
if not MODEL_DIR.is_absolute():
    MODEL_DIR = (ROOT_DIR / MODEL_DIR).resolve()

LLM_FN = os.getenv("LLM_FN", "qwen3_asr_llm.q5_k.gguf")
ENCODER_FRONTEND_FN = os.getenv("ENCODER_FRONTEND_FN", "qwen3_asr_encoder_frontend.int8.onnx")
ENCODER_BACKEND_FN = os.getenv("ENCODER_BACKEND_FN", "qwen3_asr_encoder_backend.int8.onnx")
# Load ForcedAligner + offline JSON timestamps. WebSocket finals use WS_FINAL_ALIGN.
ENABLE_ALIGNER = _safe_bool("ENABLE_ALIGNER", "true")
ALIGNER_LLM_FN = os.getenv("ALIGNER_LLM_FN", "qwen3_aligner_llm.q5_k.gguf")
ALIGNER_FRONTEND_FN = os.getenv("ALIGNER_FRONTEND_FN", "qwen3_aligner_encoder_frontend.int8.onnx")
ALIGNER_BACKEND_FN = os.getenv("ALIGNER_BACKEND_FN", "qwen3_aligner_encoder_backend.int8.onnx")
ONNX_PROVIDER = os.getenv("ONNX_PROVIDER", "DML").upper()
LLM_USE_GPU = _safe_bool("LLM_USE_GPU", "true")
# llama.cpp Vulkan reads this from the process env (not a Python API).
GGML_VK_DISABLE_F16 = _safe_bool("GGML_VK_DISABLE_F16", "false")
ASR_TEMPERATURE = _safe_float("ASR_TEMPERATURE", "0.2")  # legacy, kept for backward compat
WS_ASR_TEMPERATURE = _safe_float("WS_ASR_TEMPERATURE", "0.0")
OFFLINE_ASR_TEMPERATURE = _safe_float("OFFLINE_ASR_TEMPERATURE", "0.2")


def apply_llm_device_env() -> None:
    """Push decoder device flags into the env before llama.cpp loads DLLs."""
    if LLM_USE_GPU:
        os.environ.setdefault("GGML_VULKAN", "1")
        if GGML_VK_DISABLE_F16:
            os.environ["GGML_VK_DISABLE_F16"] = "1"
        else:
            os.environ.pop("GGML_VK_DISABLE_F16", None)
    else:
        os.environ["GGML_VULKAN"] = "0"


apply_llm_device_env()

# WS final ForcedAligner / chars. false skips align on realtime finals (faster, no chars).
# Requires ENABLE_ALIGNER=true or the aligner is not loaded.
WS_FINAL_ALIGN = _safe_bool("WS_FINAL_ALIGN", "true")
ASR_REMOVE_FILLERS = _safe_bool("ASR_REMOVE_FILLERS", "true")
WS_PARTIAL_SECONDS = _safe_float("WS_PARTIAL_SECONDS", "1.5")
# Maximum fresh PCM per partial after the first result. 0 sends the whole buffer.
# Adjacent partial windows do not overlap; finals still use the whole utterance.
WS_PARTIAL_WINDOW_SEC = _safe_float("WS_PARTIAL_WINDOW_SEC", "8")
WS_MAX_UTTERANCE_SECONDS = _safe_float("WS_MAX_UTTERANCE_SECONDS", "20")
WS_HARD_CUT_SEARCH_SECONDS = _safe_float("WS_HARD_CUT_SEARCH_SECONDS", "14")
WS_HARD_CUT_MIN_SILENCE_MS = _safe_int("WS_HARD_CUT_MIN_SILENCE_MS", "300")
WS_HARD_CUT_OVERLAP_SECONDS = _safe_float("WS_HARD_CUT_OVERLAP_SECONDS", "0.8")
WS_VAD_PRE_ROLL_MS = _safe_int("WS_VAD_PRE_ROLL_MS", "800")
WS_VAD_HANGOVER_MS = _safe_int("WS_VAD_HANGOVER_MS", "800")
WS_FLUSH_SILENCE_MS = _safe_int("WS_FLUSH_SILENCE_MS", "600")
WS_EARLY_SILENCE_MS = _safe_int("WS_EARLY_SILENCE_MS", "1000")
WS_COMMITTED_SPEECH_SECONDS = _safe_float("WS_COMMITTED_SPEECH_SECONDS", "0.8")
# Extra silence after VAD end before forcing a cut when the last partial has no 。！？
WS_VAD_END_MAX_SILENCE_MS = _safe_int("WS_VAD_END_MAX_SILENCE_MS", "1200")
# Long-utterance soft cut: split at a longer natural pause, with punctuation as a useful but optional cue.
WS_SOFT_CUT_ENABLED = _safe_bool("WS_SOFT_CUT_ENABLED", "true")
WS_SOFT_CUT_START_SECONDS = _safe_float("WS_SOFT_CUT_START_SECONDS", "12.0")
WS_SOFT_CUT_SILENCE_MS = _safe_int("WS_SOFT_CUT_SILENCE_MS", "500")
# How far before the detected silence start the next utterance may look back.
# Actual lookback is also capped by resumed_after_silence_ms / available PCM.
WS_SOFT_CUT_LOOKBACK_MS = _safe_int("WS_SOFT_CUT_LOOKBACK_MS", "200")
REALTIME_CHUNK_SIZE_SEC = _safe_float("REALTIME_CHUNK_SIZE_SEC", "8")
FILE_CHUNK_SIZE_SEC = _safe_float("FILE_CHUNK_SIZE_SEC", "40")
# Realtime encoder window. false = one growing-buffer chunk.
# true = fixed REALTIME_CHUNK_SIZE_SEC slices. Needed for shared-utt reuse.
WS_FIXED_ENCODER_WINDOW = _safe_bool("WS_FIXED_ENCODER_WINDOW", "false")
# Shared utterance state. Off keeps the 4s sliding partial path.
WS_SHARED_UTT = _safe_bool("WS_SHARED_UTT", "false")
# Snapshot the latest partial when the worker is free. Off submits immediately.
WS_LAZY_PARTIAL = _safe_bool("WS_LAZY_PARTIAL", "false")
# After a full encoder window, let the next utterance's partial run. Requires WS_SHARED_UTT.
WS_FINAL_YIELD = _safe_bool("WS_FINAL_YIELD", "false")
# Seconds of idle before unloading. 0 = keep weights loaded (desktop default).
UNLOAD_GRACE_S = _safe_int("UNLOAD_GRACE_S", "0")
REQUEST_TIMEOUT = _safe_int("REQUEST_TIMEOUT", "300")
PORT = _safe_int("PORT", "8765")
HOST = os.getenv("HOST", "127.0.0.1")
ASR_USE_SERVER_VAD = _safe_bool("ASR_USE_SERVER_VAD", "true")
FORMAT_NUM = _safe_bool("FORMAT_NUM", "true")


def _under_model_dir(env_name: str, default_fn: str) -> Path:
    """Resolve a model file: absolute as-is, otherwise relative to MODEL_DIR.

    Old values like ``./models/silero_vad.onnx`` still work (leading ``models/`` is stripped).
    """
    raw = os.getenv(env_name, default_fn).strip() or default_fn
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    parts = list(path.parts)
    if parts and parts[0] == ".":
        parts = parts[1:]
    if parts and parts[0] == "models":
        parts = parts[1:]
    if not parts:
        parts = [default_fn]
    return (MODEL_DIR.joinpath(*parts)).resolve()


VAD_MODEL_PATH = _under_model_dir("VAD_MODEL_PATH", "silero_vad.onnx")
TARGET_SR = 16000

# -----------------
# Hotword correction (phoneme-based, after ITN)
# -----------------
HOTWORD_ENABLED = _safe_bool("HOTWORD_ENABLED", "true")
HOTWORDS_PATH = Path(os.getenv("HOTWORDS_PATH", str(ROOT_DIR / "hotwords.txt"))).expanduser()
if not HOTWORDS_PATH.is_absolute():
    HOTWORDS_PATH = (ROOT_DIR / HOTWORDS_PATH).resolve()
# Match threshold 0~1, higher = stricter. 0.85 balances recall/precision for CJK.
HOTWORD_THRESHOLD = _safe_float("HOTWORD_THRESHOLD", "0.85")

DML_PAD_TO = _safe_int("DML_PAD_TO", str(int(REALTIME_CHUNK_SIZE_SEC)))

# Offline transcription result local saving.
# OFFLINE_SAVE_RESULT: when true, write the final JSON to OUTPUT_DIR/<task_id>/<stem>.json.
# When false, the result is only kept in memory (task.result) and returned via the API.
OFFLINE_SAVE_RESULT = _safe_bool("OFFLINE_SAVE_RESULT", "true")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(ROOT_DIR / "outputs"))).expanduser()
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = (ROOT_DIR / OUTPUT_DIR).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Log file directory. logger.py reads LOG_DIR directly from env (it loads before
# this module finishes), so this definition mirrors the same value for reference.
LOG_DIR = Path(os.getenv("LOG_DIR", str(default_log_dir()))).expanduser()
if not LOG_DIR.is_absolute():
    LOG_DIR = (ROOT_DIR / LOG_DIR).resolve()

OFFLINE_TARGET_CHUNK_SECONDS = _safe_float("OFFLINE_TARGET_CHUNK_SECONDS", "35")
OFFLINE_SPLIT_SEARCH_SECONDS = _safe_float("OFFLINE_SPLIT_SEARCH_SECONDS", "5")
OFFLINE_MIN_SILENCE_MS = _safe_int("OFFLINE_MIN_SILENCE_MS", "400")
OFFLINE_FALLBACK_SILENCE_MS = _safe_int("OFFLINE_FALLBACK_SILENCE_MS", "200")
OFFLINE_HARD_CUT_OVERLAP_SECONDS = _safe_float("OFFLINE_HARD_CUT_OVERLAP_SECONDS", "1.5")
OFFLINE_SILENCE_OVERLAP_SECONDS = _safe_float("OFFLINE_SILENCE_OVERLAP_SECONDS", "0.4")
OFFLINE_CUT_EPS_SECONDS = _safe_float("OFFLINE_CUT_EPS_SECONDS", "0.15")


def validate_env() -> None:
    """Validate critical environment variables at startup. Exits on invalid config."""
    errors = []

    if ONNX_PROVIDER not in _VALID_ONNX_PROVIDERS:
        errors.append(f"ONNX_PROVIDER must be one of {_VALID_ONNX_PROVIDERS}, got '{ONNX_PROVIDER}'")

    if REQUEST_TIMEOUT <= 0:
        errors.append(f"REQUEST_TIMEOUT must be positive, got {REQUEST_TIMEOUT}")
    if UNLOAD_GRACE_S < 0:
        errors.append(f"UNLOAD_GRACE_S must be non-negative, got {UNLOAD_GRACE_S}")
    if WS_PARTIAL_SECONDS <= 0:
        errors.append(f"WS_PARTIAL_SECONDS must be positive, got {WS_PARTIAL_SECONDS}")
    if WS_PARTIAL_WINDOW_SEC < 0:
        errors.append(f"WS_PARTIAL_WINDOW_SEC must be non-negative, got {WS_PARTIAL_WINDOW_SEC}")
    if WS_MAX_UTTERANCE_SECONDS <= 0:
        errors.append(f"WS_MAX_UTTERANCE_SECONDS must be positive, got {WS_MAX_UTTERANCE_SECONDS}")
    if WS_HARD_CUT_SEARCH_SECONDS <= 0:
        errors.append(f"WS_HARD_CUT_SEARCH_SECONDS must be positive, got {WS_HARD_CUT_SEARCH_SECONDS}")
    if WS_HARD_CUT_MIN_SILENCE_MS <= 0:
        errors.append(f"WS_HARD_CUT_MIN_SILENCE_MS must be positive, got {WS_HARD_CUT_MIN_SILENCE_MS}")
    if WS_HARD_CUT_OVERLAP_SECONDS < 0:
        errors.append(f"WS_HARD_CUT_OVERLAP_SECONDS must be non-negative, got {WS_HARD_CUT_OVERLAP_SECONDS}")
    if WS_VAD_PRE_ROLL_MS < 0:
        errors.append(f"WS_VAD_PRE_ROLL_MS must be non-negative, got {WS_VAD_PRE_ROLL_MS}")
    if WS_VAD_HANGOVER_MS < 0:
        errors.append(f"WS_VAD_HANGOVER_MS must be non-negative, got {WS_VAD_HANGOVER_MS}")
    if WS_FLUSH_SILENCE_MS <= 0:
        errors.append(f"WS_FLUSH_SILENCE_MS must be positive, got {WS_FLUSH_SILENCE_MS}")
    if WS_EARLY_SILENCE_MS <= 0:
        errors.append(f"WS_EARLY_SILENCE_MS must be positive, got {WS_EARLY_SILENCE_MS}")
    if WS_VAD_END_MAX_SILENCE_MS <= 0:
        errors.append(f"WS_VAD_END_MAX_SILENCE_MS must be positive, got {WS_VAD_END_MAX_SILENCE_MS}")
    if WS_COMMITTED_SPEECH_SECONDS <= 0:
        errors.append(f"WS_COMMITTED_SPEECH_SECONDS must be positive, got {WS_COMMITTED_SPEECH_SECONDS}")
    if WS_SOFT_CUT_START_SECONDS <= 0:
        errors.append(f"WS_SOFT_CUT_START_SECONDS must be positive, got {WS_SOFT_CUT_START_SECONDS}")
    if WS_SOFT_CUT_START_SECONDS >= WS_MAX_UTTERANCE_SECONDS:
        errors.append(
            f"WS_SOFT_CUT_START_SECONDS ({WS_SOFT_CUT_START_SECONDS}) must be < "
            f"WS_MAX_UTTERANCE_SECONDS ({WS_MAX_UTTERANCE_SECONDS})"
        )
    if WS_SOFT_CUT_SILENCE_MS <= 0:
        errors.append(f"WS_SOFT_CUT_SILENCE_MS must be positive, got {WS_SOFT_CUT_SILENCE_MS}")
    if WS_SOFT_CUT_SILENCE_MS >= WS_FLUSH_SILENCE_MS:
        errors.append(
            f"WS_SOFT_CUT_SILENCE_MS ({WS_SOFT_CUT_SILENCE_MS}) must be < "
            f"WS_FLUSH_SILENCE_MS ({WS_FLUSH_SILENCE_MS})"
        )
    if WS_SOFT_CUT_LOOKBACK_MS < 0:
        errors.append(f"WS_SOFT_CUT_LOOKBACK_MS must be non-negative, got {WS_SOFT_CUT_LOOKBACK_MS}")
    if REALTIME_CHUNK_SIZE_SEC <= 0:
        errors.append(f"REALTIME_CHUNK_SIZE_SEC must be positive, got {REALTIME_CHUNK_SIZE_SEC}")
    if WS_FINAL_YIELD and not WS_SHARED_UTT:
        errors.append("WS_FINAL_YIELD requires WS_SHARED_UTT=true")
    if FILE_CHUNK_SIZE_SEC <= 0:
        errors.append(f"FILE_CHUNK_SIZE_SEC must be positive, got {FILE_CHUNK_SIZE_SEC}")
    if OFFLINE_TARGET_CHUNK_SECONDS <= 0:
        errors.append(f"OFFLINE_TARGET_CHUNK_SECONDS must be positive, got {OFFLINE_TARGET_CHUNK_SECONDS}")
    if OFFLINE_SPLIT_SEARCH_SECONDS <= 0:
        errors.append(f"OFFLINE_SPLIT_SEARCH_SECONDS must be positive, got {OFFLINE_SPLIT_SEARCH_SECONDS}")
    if OFFLINE_TARGET_CHUNK_SECONDS >= FILE_CHUNK_SIZE_SEC:
        errors.append(
            f"OFFLINE_TARGET_CHUNK_SECONDS ({OFFLINE_TARGET_CHUNK_SECONDS}) must be < FILE_CHUNK_SIZE_SEC ({FILE_CHUNK_SIZE_SEC})"
        )
    if OFFLINE_HARD_CUT_OVERLAP_SECONDS < 0:
        errors.append(f"OFFLINE_HARD_CUT_OVERLAP_SECONDS must be non-negative, got {OFFLINE_HARD_CUT_OVERLAP_SECONDS}")
    if OFFLINE_HARD_CUT_OVERLAP_SECONDS >= FILE_CHUNK_SIZE_SEC:
        errors.append(
            f"OFFLINE_HARD_CUT_OVERLAP_SECONDS ({OFFLINE_HARD_CUT_OVERLAP_SECONDS}) must be < FILE_CHUNK_SIZE_SEC ({FILE_CHUNK_SIZE_SEC})"
        )
    if OFFLINE_SILENCE_OVERLAP_SECONDS < 0:
        errors.append(f"OFFLINE_SILENCE_OVERLAP_SECONDS must be non-negative, got {OFFLINE_SILENCE_OVERLAP_SECONDS}")
    if OFFLINE_SILENCE_OVERLAP_SECONDS >= FILE_CHUNK_SIZE_SEC:
        errors.append(
            f"OFFLINE_SILENCE_OVERLAP_SECONDS ({OFFLINE_SILENCE_OVERLAP_SECONDS}) must be < FILE_CHUNK_SIZE_SEC ({FILE_CHUNK_SIZE_SEC})"
        )
    if OFFLINE_MIN_SILENCE_MS <= 0:
        errors.append(f"OFFLINE_MIN_SILENCE_MS must be positive, got {OFFLINE_MIN_SILENCE_MS}")
    if OFFLINE_FALLBACK_SILENCE_MS <= 0:
        errors.append(f"OFFLINE_FALLBACK_SILENCE_MS must be positive, got {OFFLINE_FALLBACK_SILENCE_MS}")
    if OFFLINE_CUT_EPS_SECONDS < 0:
        errors.append(f"OFFLINE_CUT_EPS_SECONDS must be non-negative, got {OFFLINE_CUT_EPS_SECONDS}")

    log_level = os.getenv("LOG_LEVEL", "info").upper()
    log_level = _LOG_LEVEL_ALIASES.get(log_level, log_level)
    if log_level not in _VALID_LOG_LEVELS:
        errors.append(f"LOG_LEVEL must be one of {_VALID_LOG_LEVELS}, got '{log_level}'")

    required_files = [
        MODEL_DIR / LLM_FN,
        MODEL_DIR / ENCODER_FRONTEND_FN,
        MODEL_DIR / ENCODER_BACKEND_FN,
        VAD_MODEL_PATH,
    ]
    if ENABLE_ALIGNER:
        required_files.extend([
            MODEL_DIR / ALIGNER_LLM_FN,
            MODEL_DIR / ALIGNER_FRONTEND_FN,
            MODEL_DIR / ALIGNER_BACKEND_FN,
        ])
    for path in required_files:
        if not path.exists():
            errors.append(f"Missing model file: {path}")

    if errors:
        for err in errors:
            log.error("Config validation failed: {}", err)
        sys.exit(1)

    log.info("Config validation passed | model_dir={} provider={}", MODEL_DIR, ONNX_PROVIDER)
