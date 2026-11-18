# ASR 托盘启动器

双击项目根目录的 `ASR.exe`。它使用同目录的 `.env`、`run.py` 和现有模型，后台启动服务。不要单独移动这个 exe。

右下角托盘图标名称为 ASR，悬停显示服务状态；右键菜单中的“退出”同时关闭服务。Windows 可能把图标收进托盘的向上箭头菜单，可手动拖到常驻区域。

源码入口：`src/tray_app.py`。

启动器内置托盘所需组件，ASR 服务仍使用独立 Python 环境。优先支持便携包的 `runtime/python.exe + app/run.py`；项目目录可通过 `.env` 指定已安装项目依赖的 Python：

```dotenv
ASR_PYTHON=C:/path/to/env/python.exe
```

也会查找 `.venv`、常用位置的 conda lingting 环境以及 PATH 中的 Python。未找到环境时，托盘显示异常信息。服务输出保存在 `.env` 的 `LOG_DIR` 下的 `service.log`。

重新构建（先在构建环境安装 `packaging/requirements-build.txt`）：

```powershell
powershell -ExecutionPolicy Bypass -File packaging/build-launcher.ps1 -Python C:/path/to/build/python.exe
```

完整便携包构建仍使用 `packaging/build.ps1`；本次启动器构建只生成项目根目录的 `ASR.exe`。
