# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Windows-local FastAPI server for Qwen3-ASR using **ONNX Encoder (DirectML) + GGUF Decoder (llama.cpp / Vulkan)**. No Docker/PyTorch in the supported runtime. See `代码改造计划.md`.

## Environment

Use conda env **`lingting`** (`D:\anaconda3\envs\lingting`, Python 3.10). Do not run with `base` / other envs.

```bash
conda activate lingting
```

## Common Commands

```bash
conda activate lingting

# Install (no torch)
pip install -r requirements.txt

# Start locally
python run.py

# Green pack (do not zip this git repo): packaging/build.ps1 → dist/ASR/
# See packaging/README.md

# Health check
curl http://127.0.0.1:8765/health

# Test transcription
curl -X POST http://127.0.0.1:8765/offline/transcribe-path -H "Content-Type: application/json" -d "{\"audio_path\":\"D:\\\\audio\\\\sample.wav\"}"
```

Place llama.cpp Vulkan DLLs in `vendor/qwen_asr_gguf/inference/bin/` and models in `models/` (`qwen3_asr_llm.q5_k.gguf` + int8 encoder ONNX + `silero_vad.onnx`).

### Manual clients (E2Etest)

Server must be running. Set `WAV_PATH` at the top of each script, then:

```bash
conda activate lingting
python E2Etest/test_offline_split.py
python E2Etest/test_api_http.py
python E2Etest/test_websocket.py
```

## Architecture

### File Organization

- `src/server.py` — Core FastAPI server with inference logic, priority queue, WebSocket handling (~1300 lines)
- `src/offline_tasks.py` — In-memory offline HTTP task store, queue submit, JSON output
- `src/offline_split.py` — Silence-first chunk planner and Aligner timestamp dedup
- `src/segments.py` — Aligner tokens → sentence segments for offline JSON
- `src/gateway.py` — Gateway proxy mode (GATEWAY_MODE=true); routes to worker subprocess
- `src/worker.py` — Inference worker for gateway mode; imports logic from server.py
- `src/subtitle.py` — Subtitle generation module: ForcedAligner, segmentation, SRT formatting
- `src/logger.py` — Loguru-based structured logging with uvicorn/FastAPI interception
- `src/schemas.py` — Pydantic models for Swagger UI documentation
- `src/export_onnx.py` — Export encoder to ONNX for ORT acceleration
- `src/build_trt.py` — Build TensorRT engine for encoder
- `E2Etest/` — live-server E2E: `test_api_http.py`, `test_websocket.py`

### API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Health check, returns model status and config |
| `/offline/transcribe` | POST | Upload audio, returns `task_id` |
| `/offline/transcribe-path` | POST | Local `audio_path`, returns `task_id` |
| `/offline/tasks/{id}` | GET | Offline task status |
| `/offline/tasks/{id}/result` | GET | `{segments: [...]}` when completed |
| `/realtime/stream` | WebSocket | Real-time streaming with raw PCM input |

### Concurrency Model

**PriorityInferQueue** (replaces simple semaphore):
- Min-heap priority queue with `priority: int` (lower = higher priority)
- WebSocket requests use priority=0, HTTP uploads use priority=1
- This prevents long file uploads from blocking real-time WebSocket transcription
- Single dedicated ThreadPoolExecutor (`max_workers=1`) for all GPU inference
- All inference runs via `run_in_executor()` to keep async event loop unblocked

### Model Lifecycle

- Model loads on first request (not at startup in standalone mode)
- `_idle_watchdog()` background task unloads after `IDLE_TIMEOUT` seconds (default 120s, 0 = disabled)
- `asyncio.Lock()` prevents load/unload race conditions
- GPU memory explicitly released via `release_gpu_memory()` after operations

**Gateway Mode** (`GATEWAY_MODE=true`):
- Splits into gateway (port 8000) + worker subprocess (port 8001)
- Gateway proxies all requests to worker via HTTP/WebSocket
- Killing worker process reclaims ALL RAM/VRAM (useful for memory leak scenarios)

### WebSocket Real-Time Transcription (`/realtime/stream`)

- Accepts raw PCM: 16-bit little-endian, 16kHz, mono
- Accumulates audio in a sliding window (up to `WS_WINDOW_MAX_S` seconds, default 6s)
- **Sliding window**: Re-transcribes entire accumulated audio each trigger for full context; partials are cumulative (client replaces, never appends)
- **Silence padding**: 600ms silence appended on `flush` command to commit trailing words (`WS_FLUSH_SILENCE_MS`)
- **VAD gating**: Silero VAD skips inference for silent frames; auto-flushes on speech→silence transitions (`ASR_USE_SERVER_VAD=true` default, overridable per-connection via query param or config action)
- **Dual-model**: If `DUAL_MODEL=true`, uses 0.6B for partials, 1.7B for final transcription
- Control messages: `flush`, `reset`, `config` (set language, toggle `use_server_vad`)
- Buffer transcribed on disconnect (no audio loss)

### Audio Preprocessing & Chunking

Handled natively by the SDK's `model.transcribe()` — mono conversion, resampling to 16kHz, float32 normalization, and long-audio chunking (up to 20min) are all internal. server.py only adds server-level concerns: priority queue, WebSocket streaming, GPU optimizations, idle lifecycle.

**Lazy imports:** Heavy libraries (torch, soundfile, qwen_asr) are imported on first request, not at module load. Idle container RAM is ~50-100MB instead of ~2.4GB.

### Optimizations (Opt-in)

All Phase 3 features are gated behind environment variables — safe to experiment without breaking defaults.

| Feature | Env Var | Description |
|---------|---------|-------------|
| Flash Attention 2 | auto-detected | Falls back to SDPA if unavailable |
| Pinned memory | auto | Pre-allocated 30s buffer for fast CPU→GPU transfer |
| CUDA streams | auto | Async DMA pipeline for transfer/compute overlap |
| INT8 quantization | `QUANTIZE=int8` | bitsandbytes W8A8 (~50% VRAM reduction) |
| FP8 quantization | `QUANTIZE=fp8` | torchao (requires sm_89+ Hopper/Ada) |
| Speculative decoding | `USE_SPECULATIVE=true` | 0.6B draft + 1.7B verifier (~2x speed) |
| ONNX encoder | `ONNX_ENCODER_PATH` | ORT-accelerated encoder forward pass |
| TensorRT encoder | `TRT_ENCODER_PATH` | Compiled TRT engine for encoder |
| CUDA Graphs warmup | `USE_CUDA_GRAPHS=true` | 3 extra warmup passes for kernel caching |
| NUMA CPU pinning | `NUMA_NODE=0` | Pin to GPU-collocated NUMA node |
| Granian ASGI | `USE_GRANIAN=true` | Rust-based ASGI server (alternative to uvicorn) |

## Key Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_ID` | `Qwen/Qwen3-ASR-1.7B` | HuggingFace model (0.6B for speed, 1.7B for accuracy) |
| `FAST_MODEL_ID` | `Qwen/Qwen3-ASR-0.6B` | Draft/partial model for speculative/DUAL_MODE |
| `IDLE_TIMEOUT` | `120` | Seconds before model unloads (0 = keep loaded) |
| `LOG_LEVEL` | `INFO` | Log verbosity (DEBUG, INFO, WARNING, ERROR) |
| `REQUEST_TIMEOUT` | `300` | Max inference time per request |
| `WS_BUFFER_SIZE` | `14400` | WebSocket audio buffer (~450ms at 16kHz) |
| `WS_WINDOW_MAX_S` | `6.0` | Max seconds of audio in sliding window for WS streaming |
| `WS_FLUSH_SILENCE_MS` | `600` | Silence padding on flush (ms) |
| `ASR_USE_SERVER_VAD` | `true` | Server-side VAD: auto-flush + silence skip (overridable per-connection) |
| `GATEWAY_MODE` | `false` | Run as gateway+worker split |
| `DUAL_MODEL` | `false` | Load both 0.6B and 1.7B models |
| `QUANTIZE` | `""` | `int8` or `fp8` |
| `FORCED_ALIGNER_ID` | `Qwen/Qwen3-ForcedAligner-0.6B` | HuggingFace model for word-level alignment |

Port mapping: container 8000 → host 8100.

### Docker Build

Base image: `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel`. The `devel` variant is required because `flash-attn` builds from source and needs `nvcc`. HuggingFace model cache is persisted via `./models` volume mount.

## Docs

- `docs/WEBSOCKET_USAGE.md` — WebSocket protocol, connection format, example Python client
- `docs/GRANIAN_BENCHMARK.md` — Performance comparison of ASGI servers
- `ROADMAP.md` — Milestone planning (8 phases completed, backlog)
- `CHANGELOG.md` — Version history
- `LEARNING_LOG.md` — Technical learnings and decisions
