# 绿色版 ASR

不要压缩 git 仓库 `stt-qwen3`。给别人的是 `packaging/build.ps1` 生成的 `dist/ASR/`。

## 构建（开发机，conda `lingting`）

```powershell
conda activate lingting
pip install -r packaging/requirements-build.txt
powershell -ExecutionPolicy Bypass -File packaging/build.ps1
```

产物：`dist/ASR/`（不含 GGUF/ONNX 模型）。其中 `.env` 直接复制仓库根目录的开发环境 `.env`，打包脚本不再另写一份。

## 发给别人

1. 把 `dist/ASR` 打成 zip。
2. 对方解压后，把模型文件放到 `models/`（与开发机 `models/` 相同的 GGUF + encoder ONNX + `silero_vad.onnx`）。
3. 双击 `ASRTray.exe`。不需要安装 Python / Anaconda。

## 目录

```
ASR/
  ASRTray.exe
  runtime/          嵌入式 Python + site-packages
  app/              run.py + src + vendor（可单独替换升级 ASR）
  models/
  .env
  logs/
  outputs/
```

升级转写逻辑：替换 `app/` 下的 `.py`，一般不用重打 Tray。
