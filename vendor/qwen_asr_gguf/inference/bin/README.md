# llama.cpp Windows Vulkan DLLs

Place the following files here (same directory as this README):

- `llama.dll`
- `ggml.dll`
- `ggml-base.dll`
- `ggml-vulkan.dll`
- all `ggml-cpu-*.dll` and other `ggml-*.dll` from the same zip
- `libomp.dll` (OpenMP runtime from that zip)

Download: [llama.cpp Releases](https://github.com/ggml-org/llama.cpp/releases)
Recommended build recorded in `VENDOR.md`: `llama-b10901-bin-win-vulkan-x64.zip`

`llama.py` loads DLLs from **this** directory, not `qwen_asr_gguf/bin/`.
Do not mix DLLs from different builds.
