@echo off
set SRC=e:\cursorProject\stt-qwen3\vendor\qwen_asr_gguf\inference\bin-b10901-orig
set DST=e:\cursorProject\stt-qwen3\vendor\qwen_asr_gguf\inference\bin

echo === Removing patched DLLs from vendor bin ===
del /q "%DST%\ggml*.dll" 2>nul
del /q "%DST%\llama*.dll" 2>nul
del /q "%DST%\mtmd.dll" 2>nul

echo === Restoring official DLLs from backup ===
copy /y "%SRC%\*.dll" "%DST%\"

echo === Done ===
dir /b "%DST%\*.dll"
