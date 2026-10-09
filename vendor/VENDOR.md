# Vendor: qwen_asr_gguf

Source: https://github.com/HaujetZhao/Qwen3-ASR-GGUF
Copied: `qwen_asr_gguf/` inference runtime (no export scripts, no CLI).

## Local patches

1. `inference/__init__.py` — drop `exporters` imports.
2. `inference/asr.py` — `streaming=False` on `asr()` / `_decode` / `_safe_decode`; `do_align`; add `close()` (including aligner).
3. `inference/encoder.py` — add `close()` for ONNX sessions.
4. `inference/llama.py` — align `llama_model_params` / `llama_context_params` with llama.cpp **b10901** (`load_mode`, `lazy_mode`, `load_mtp`, `n_outputs_max_per_seq`). Mismatched ABI crashes in `llama_init_from_model`.
5. `inference/aligner.py` — vendored ForcedAligner; add `close()`.
6. `inference/chinese_itn.py` — removed; ITN lives in `src/chinese_itn/`. Dropped `itn` re-export from `inference/__init__.py`.

## llama.cpp

`inference/bin/` must contain the win-vulkan DLLs from **one** build. See `inference/bin/README.md`.
Unpacked: `b10901` (`llama-b10901-bin-win-vulkan-x64.zip`) into `inference/bin/`.
Do not mix DLLs from different builds (ABI changed between `b9977` and `b10901`; mismatched params crash in `llama_init_from_model`).
Official `b10901` DLLs backed up at `inference/bin-b10901-orig/` (restorable via `inference/restore_official_dlls.bat`).
