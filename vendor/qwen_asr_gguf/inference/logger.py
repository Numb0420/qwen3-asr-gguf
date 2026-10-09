try:
    from .. import logger
except Exception:
    import logging
    logger = logging.getLogger("qwen_asr_gguf")
