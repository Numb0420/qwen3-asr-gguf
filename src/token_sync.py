"""Sync formatted text (ITN / punct) back onto WordTimestamp tokens.

Adapted from CapsWriter-Offline core/tools/token_sync.py (MIT, Haujet Zhao):
expand multi-char tokens, SequenceMatcher against the formatted string, then
merge ASCII alnum runs. This project keeps start/end instead of a single stamp.
"""
from __future__ import annotations

import difflib

from backend.base import WordTimestamp


def sync_words_from_text(
    words: list[WordTimestamp] | None,
    formatted_text: str,
) -> list[WordTimestamp]:
    items = list(words or [])
    if not items:
        return items
    raw = "".join(w.word or "" for w in items)
    if not formatted_text or formatted_text == raw:
        return items

    need_merge = any(len(w.word or "") > 1 for w in items)
    work = _expand_words(items) if need_merge else items
    work_text = "".join(w.word or "" for w in work)
    if formatted_text == work_text:
        return items

    char_to_tok: list[int] = []
    for idx, token in enumerate(work):
        char_to_tok.extend([idx] * max(1, len(token.word or "")))

    sm = difflib.SequenceMatcher(None, work_text, formatted_text)
    new_words: list[WordTimestamp] = []
    emitted: set[int] = set()

    for op, ri1, ri2, fi1, fi2 in sm.get_opcodes():
        if op == "equal":
            _handle_equal(work, char_to_tok, ri1, ri2, new_words, emitted)
        elif op == "insert":
            _handle_insert(formatted_text, fi1, fi2, new_words, work)
        elif op == "delete":
            _mark_range(char_to_tok, ri1, ri2, emitted)
        elif op == "replace":
            _handle_replace(work, char_to_tok, formatted_text, fi1, fi2, ri1, ri2, new_words, emitted)

    if need_merge:
        new_words = _merge_ascii_words(new_words)
    return new_words


def _expand_words(words: list[WordTimestamp]) -> list[WordTimestamp]:
    flat: list[WordTimestamp] = []
    for w in words:
        token = w.word or ""
        if len(token) <= 1:
            flat.append(w)
            continue
        for ch in token:
            flat.append(WordTimestamp(word=ch, start=w.start, end=w.end))
    return flat


def _merge_ascii_words(words: list[WordTimestamp]) -> list[WordTimestamp]:
    merged: list[WordTimestamp] = []
    buf: list[str] = []
    start = 0.0
    end = 0.0

    def flush() -> None:
        nonlocal buf, start, end
        if buf:
            merged.append(WordTimestamp(word="".join(buf), start=start, end=end))
            buf = []

    for w in words:
        token = w.word or ""
        if token.isascii() and token.isalnum():
            if not buf:
                start = float(w.start)
            buf.append(token)
            end = float(w.end)
        else:
            flush()
            merged.append(w)
    flush()
    return merged


def _tokenize_replacement(text: str) -> list[str]:
    tokens: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            tokens.append("".join(buf))
            buf.clear()

    for ch in text:
        if ch.isascii() and ch.isalnum():
            buf.append(ch)
        elif ch.isalnum():
            flush()
            tokens.append(ch)
        else:
            flush()
            tokens.append(ch)
    flush()
    return tokens


def _handle_equal(
    work: list[WordTimestamp],
    char_to_tok: list[int],
    ri1: int,
    ri2: int,
    new_words: list[WordTimestamp],
    emitted: set[int],
) -> None:
    for ri in range(ri1, ri2):
        if ri >= len(char_to_tok):
            continue
        ti = char_to_tok[ri]
        if ti not in emitted:
            new_words.append(work[ti])
            emitted.add(ti)


def _handle_insert(
    formatted_text: str,
    fi1: int,
    fi2: int,
    new_words: list[WordTimestamp],
    work: list[WordTimestamp],
) -> None:
    text = formatted_text[fi1:fi2]
    if not text:
        return
    if new_words:
        ts_start = float(new_words[-1].end)
        ts_end = ts_start
    elif work:
        ts_start = float(work[0].start)
        ts_end = ts_start
    else:
        ts_start = 0.0
        ts_end = 0.0
    for token in _tokenize_replacement(text):
        new_words.append(WordTimestamp(word=token, start=ts_start, end=ts_end))


def _mark_range(char_to_tok: list[int], ri1: int, ri2: int, emitted: set[int]) -> None:
    if ri1 >= ri2 or ri1 >= len(char_to_tok):
        return
    ti_start = char_to_tok[ri1]
    ti_end = char_to_tok[min(ri2, len(char_to_tok)) - 1] + 1
    for ti in range(ti_start, ti_end):
        emitted.add(ti)


def _handle_replace(
    work: list[WordTimestamp],
    char_to_tok: list[int],
    formatted_text: str,
    fi1: int,
    fi2: int,
    ri1: int,
    ri2: int,
    new_words: list[WordTimestamp],
    emitted: set[int],
) -> None:
    if ri1 >= ri2 or ri1 >= len(char_to_tok):
        return
    ti_start = char_to_tok[ri1]
    ti_end = char_to_tok[min(ri2, len(char_to_tok)) - 1] + 1
    for ti in range(ti_start, ti_end):
        emitted.add(ti)
    replacement = formatted_text[fi1:fi2]
    if not replacement:
        return
    span_start = float(work[ti_start].start)
    span_end = float(work[ti_end - 1].end)
    for token in _tokenize_replacement(replacement):
        new_words.append(WordTimestamp(word=token, start=span_start, end=span_end))
