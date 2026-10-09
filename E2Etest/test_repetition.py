"""No-GPU checks for phrase-loop abort and official-style collapse.

    conda activate lingting
    python E2Etest/test_repetition.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
VENDOR = ROOT / "vendor"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

from repetition import collapse_repetitions
from qwen_asr_gguf.inference.repetition import (
    max_new_tokens_for_audio,
    suffix_loop_pattern,
    trim_loop_tail,
)


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_keeps_spoken_repeats() -> None:
    _assert(collapse_repetitions("没有没有没有") == "没有没有没有", "3x 没有 should stay")
    _assert(collapse_repetitions("对对对") == "对对对", "3x 对 should stay")
    _assert(collapse_repetitions("好好好") == "好好好", "3x 好 should stay")


def test_collapses_long_phrase_loop() -> None:
    loop = "发文章？对。" * 12
    out = collapse_repetitions(loop)
    _assert(out == "发文章？对。", f"expected 1 copy, got {out!r}")


def test_suffix_loop_detects_and_trims() -> None:
    pat = "发文章？对。"
    text = "前文。" + pat * 6
    found = suffix_loop_pattern(text, min_reps=6)
    _assert(found == pat, f"expected {pat!r}, got {found!r}")
    trimmed = trim_loop_tail(text, found, keep=2)
    _assert(trimmed == "前文。" + pat * 2, trimmed)


def test_suffix_loop_ignores_short_spoken() -> None:
    _assert(suffix_loop_pattern("没有没有没有", min_reps=6) is None, "3x phrase must not abort")
    _assert(suffix_loop_pattern("对对对", min_reps=6) is None, "3x char must not abort")


def test_suffix_loop_single_char_higher_threshold() -> None:
    # 6 consecutive single-char reps must NOT abort (natural spoken repetition)
    _assert(suffix_loop_pattern("前文，对对对对对对") is None, "6x single char must not abort")
    # 12 consecutive single-char reps SHOULD abort (true loop)
    _assert(suffix_loop_pattern("前文，" + "对" * 12) == "对", "12x single char must abort")
    # Multi-char phrase loops keep 6-rep threshold
    pat = "发文章？对。"
    _assert(suffix_loop_pattern("前文。" + pat * 6) == pat, "6x multi-char phrase must abort")


def test_token_budget() -> None:
    _assert(max_new_tokens_for_audio(35.0) == 280, str(max_new_tokens_for_audio(35.0)))
    _assert(max_new_tokens_for_audio(80.0) == 512, str(max_new_tokens_for_audio(80.0)))
    _assert(max_new_tokens_for_audio(1.0) == 32, str(max_new_tokens_for_audio(1.0)))


if __name__ == "__main__":
    test_keeps_spoken_repeats()
    test_collapses_long_phrase_loop()
    test_suffix_loop_detects_and_trims()
    test_suffix_loop_ignores_short_spoken()
    test_token_budget()
    print("ok")
