"""
FunASR-GGUF: 混合 ASR 推理引擎

使用 ONNX Runtime (encoder/CTC) + llama.cpp (GGUF decoder) 进行语音识别

API 兼容 sherpa-onnx，可直接替换使用。
"""

import logging
import sys
import os
from logging.handlers import TimedRotatingFileHandler

# 获取项目根目录 (适配打包环境)
if getattr(sys, 'frozen', False):
    # 打包环境：sys.executable 位于 dist/Project/ 根目录
    ROOT_DIR = os.path.dirname(sys.executable)
else:
    # 源码环境：__file__ 位于 <root>/vendor/qwen_asr_gguf/__init__.py
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _resolve_log_dir() -> str:
    """日志目录解析：优先 LOG_DIR 环境变量（相对 ROOT_DIR），否则回退默认。

    与 src/paths.py + src/logger.py 保持一致，确保 latest.log 与 server.log 同目录。
    logger.py 加载早于 config.py，这里同样直接读环境变量而不导入 config。
    """
    env_log_dir = os.getenv("LOG_DIR")
    if env_log_dir:
        p = os.path.expanduser(env_log_dir)
        if not os.path.isabs(p):
            p = os.path.join(ROOT_DIR, p)
        return os.path.normpath(p)
    # 默认与 src/paths.py default_log_dir() 非打包分支一致
    return os.path.join(ROOT_DIR, "vendor", "logs")


default_log_dir = _resolve_log_dir()
default_log_file = os.path.join(default_log_dir, "latest.log")


def setup_logging(level: int = logging.WARNING, log_file: str = default_log_file):
    """
    配置全局日志，按天滚动（午夜切割，保留 30 天归档）。

    Args:
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件名

    Returns:
        配置好的 logger 实例
    """
    # 获取根 logger
    root_logger = logging.getLogger('qwen_asr_gguf')
    root_logger.setLevel(level)  # 接收所有级别的日志
    root_logger.handlers.clear()  # 清除已有处理器

    # 文件处理器：按天滚动
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        # when='midnight' 每天 00:00 切割；backupCount=30 保留 30 个历史文件
        # 归档命名形如 latest.log.2024-09-17（suffix 默认 %Y-%m-%d）
        file_handler = TimedRotatingFileHandler(
            log_file,
            when='midnight',
            interval=1,
            backupCount=30,
            encoding='utf-8',
            delay=True,
        )
        file_handler.setLevel(level)  # 文件通常记录更详细的信息
        file_formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    return root_logger


# 初始化默认日志配置（默认 INFO 级别）
try:
    from .. import logger
except:
    logger = setup_logging(level=logging.INFO)

