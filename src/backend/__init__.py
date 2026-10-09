from .base import ASRBackend, ASRResult, BackendState, WordTimestamp

__all__ = ["ASRBackend", "ASRResult", "BackendState", "WordTimestamp", "QwenGGUFBackend"]


def __getattr__(name: str):
    if name == "QwenGGUFBackend":
        from .gguf_backend import QwenGGUFBackend
        return QwenGGUFBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
