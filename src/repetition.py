"""Repetition guards — re-export vendor implementation for server/tests."""
from pathlib import Path
import sys

_VENDOR = Path(__file__).resolve().parent.parent / "vendor"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

from qwen_asr_gguf.inference.repetition import collapse_repetitions  # noqa: E402


def carry_prefix_tail(text: str, n: int = 5) -> str:
    """Approx last n tokens for a WS hard-cut stitch: CJK by char, else whitespace words."""
    s = (text or "").strip()
    if not s or n <= 0:
        return ""
    if any("\u4e00" <= ch <= "\u9fff" for ch in s):
        return s[-n:]
    return " ".join(s.split()[-n:])
