"""Group aligner tokens into sentence segments for offline JSON."""
from __future__ import annotations

import unicodedata

from backend.base import WordTimestamp

_SENT_END = set("。！？；!?")
_PUNCT = set("。！？；!?，,、：:;；…—-\"'“”‘’（）()[]《》〈〉")


def _is_punct_token(text: str) -> bool:
    raw = (text or "").strip()
    if not raw:
        return True
    return all(ch in _PUNCT or unicodedata.category(ch).startswith("P") for ch in raw)


def _split_trailing_punct(token: str) -> tuple[str, str]:
    i = len(token)
    while i > 0:
        ch = token[i - 1]
        if ch in _PUNCT or unicodedata.category(ch).startswith("P"):
            i -= 1
            continue
        break
    return token[:i], token[i:]


def _sentence_end_punct(text: str) -> str:
    stripped = (text or "").rstrip()
    i = len(stripped)
    while i > 0 and stripped[i - 1] in _SENT_END:
        i -= 1
    return stripped[i:]


def _flush(buf_text: str, chars: list[dict], index: int) -> dict:
    text = buf_text.strip()
    start = chars[0]["start"] if chars else 0.0
    end = chars[-1]["end"] if chars else start
    return {
        "index": index,
        "start": start,
        "end": end,
        "text": text,
        "punctuation": _sentence_end_punct(text),
        "speaker": None,
        "chars": chars,
    }


def words_to_segments(words: list[WordTimestamp] | None, fallback_text: str = "") -> list[dict]:
    items = list(words or [])
    if not items:
        text = (fallback_text or "").strip()
        if not text:
            return []
        return [_flush(text, [], 1)]

    segments: list[dict] = []
    buf_text = ""
    chars: list[dict] = []
    index = 1

    def _commit() -> None:
        nonlocal buf_text, chars, index
        if buf_text.strip() or chars:
            segments.append(_flush(buf_text, chars, index))
            index += 1
        buf_text = ""
        chars = []

    for item in items:
        token = item.word or ""
        zero_dur = float(item.start) == float(item.end)
        # 0-duration gap items from the aligner (punctuation / spaces) and punct-only tokens.
        if _is_punct_token(token) or (zero_dur and any(ch in _SENT_END for ch in token)):
            buf_text += token
            if any(ch in _SENT_END for ch in token):
                _commit()
            continue

        word_part, punct_part = _split_trailing_punct(token)
        if word_part:
            chars.append({"text": word_part, "start": item.start, "end": item.end})
            buf_text += word_part
        if punct_part:
            buf_text += punct_part
            if any(ch in _SENT_END for ch in punct_part):
                _commit()

    _commit()
    return segments
