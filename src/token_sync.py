"""Sync formatted text (ITN / punct) back onto WordTimestamp tokens.

Adapted from CapsWriter-Offline core/tools/token_sync.py (MIT, Haujet Zhao):
expand multi-char tokens, SequenceMatcher against the formatted string, then
merge ASCII alnum runs and complete numbers. This project keeps start/end
instead of a single stamp.
"""
from __future__ import annotations

import difflib
import re

from backend.base import WordTimestamp

_NUMBER = re.compile(r"(?<![A-Za-z0-9])[+-]?[0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?(?![A-Za-z0-9])")


def sync_words_from_replacements(
    words: list[WordTimestamp] | None,
    original_text: str,
    replacements: list[tuple[int, int, str]],
) -> list[WordTimestamp]:
    """Map explicit original-text spans to whole replacement tokens."""
    items = list(words or [])
    if not items:
        return []
    # Stitching may leave punctuation differences between text and tokens.
    if "".join(w.word or "" for w in items) != original_text:
        items = sync_words_from_text(items, original_text)
    ranges = []
    offset = 0
    for word in items:
        stop = offset + len(word.word or "")
        ranges.append((offset, stop, word))
        offset = stop
    if offset != len(original_text):
        raise ValueError("Original text does not match timestamp tokens")

    def slice_words(start: int, end: int) -> list[WordTimestamp]:
        sliced = []
        for left, right, word in ranges:
            if left < end and right > start:
                text = word.word[max(start, left) - left:min(end, right) - left]
                sliced.append(WordTimestamp(text, word.start, word.end))
        return sliced

    output = []
    cursor = 0
    for start, end, text in sorted(replacements):
        if not (cursor <= start < end <= len(original_text)):
            raise ValueError("Invalid or overlapping hotword replacement spans")
        output.extend(slice_words(cursor, start))
        source = slice_words(start, end)
        if not source:
            raise ValueError("Hotword replacement has no source timestamps")
        output.append(WordTimestamp(text, min(w.start for w in source),
                                    max(w.end for w in source)))
        cursor = end
    output.extend(slice_words(cursor, len(original_text)))
    return output


def _merge_number_words(words: list[WordTimestamp]) -> list[WordTimestamp]:
    """Keep a formatted number together, using the union of its source times."""
    text = "".join(w.word or "" for w in words)
    boundaries = {0: 0}
    offset = 0
    for i, word in enumerate(words):
        offset += len(word.word or "")
        boundaries[offset] = i + 1
    merged: list[WordTimestamp] = []
    cursor = 0
    for match in _NUMBER.finditer(text):
        # Do not split an existing token (e.g. an identifier or a unit).
        if match.start() not in boundaries or match.end() not in boundaries:
            continue
        first, last = boundaries[match.start()], boundaries[match.end()]
        if first >= last:
            continue
        merged.extend(words[cursor:first])
        source = words[first:last]
        merged.append(WordTimestamp(
            word=match.group(), start=min(w.start for w in source),
            end=max(w.end for w in source),
        ))
        cursor = last
    merged.extend(words[cursor:])
    return merged


def sync_words_from_text(
    words: list[WordTimestamp] | None,
    formatted_text: str,
) -> list[WordTimestamp]:
    items = list(words or [])
    if not items:
        return items
    raw = "".join(w.word or "" for w in items)
    if not formatted_text:
        return items
    if formatted_text == raw:
        return _merge_number_words(items)

    need_merge = any(len(w.word or "") > 1 for w in items)
    work = _expand_words(items) if need_merge else items
    work_text = "".join(w.word or "" for w in work)
    if formatted_text == work_text:
        return _merge_number_words(items)

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
    return _merge_number_words(new_words)


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
