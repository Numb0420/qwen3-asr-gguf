"""Offline hard-cut stitch: reuse WS text dedup plus collapsed-token cleanup."""
from __future__ import annotations

import re

from backend.base import WordTimestamp
from offline_split import (
    drop_overlap_tokens,
    shift_words,
    stitch_chunk_words,
    strip_forced_cut_words,
)
from ws_overlap import (
    leftover_echo,
    restore_nonoverlap_dropped,
    strip_boundary,
    strip_forced_cut_stop,
    sync_words_to_text,
    trim_cut_fragment,
    trim_dangling_tail,
)

_LEADING_SHORT = re.compile(r"^([\u4e00-\u9fff]{1,2})([。！？!?；;])")
_COLLAPSED_EPS = 1e-6
_HEAD_COLLAPSE_WINDOW = 0.3
_SHORT_GAP_S = 2.0
_PLAIN_PUNCT = set("。！？；!?，,、：:;；…—-\"'“”‘’（）()[]《》〈〉 \t")


def _join_words(words: list[WordTimestamp] | None) -> str:
    return "".join(w.word for w in (words or []) if w.word)


def _plain(text: str) -> str:
    return "".join(ch for ch in (text or "") if ch not in _PLAIN_PUNCT)


def _is_collapsed(word: WordTimestamp) -> bool:
    return abs(float(word.end) - float(word.start)) <= _COLLAPSED_EPS


def _collapsed_is_overlap(run_text: str, prev_text: str, follow_text: str) -> bool:
    """True only for cut-boundary echoes, not real words that happen to have 0 duration."""
    run = run_text or ""
    if run and follow_text.startswith(run):
        return True
    if not prev_text or not run:
        return False
    if leftover_echo(prev_text, run):
        return True
    if strip_boundary(prev_text, run, acoustic_overlap=True) != run:
        return True
    prev_core = _plain(prev_text)
    run_core = _plain(run)
    max_n = min(len(prev_core), len(run_core), 16)
    for n in range(max_n, 3, -1):
        if prev_core[-n:] and prev_core[-n:] in run_core:
            return True
    return False


def drop_collapsed_cut_tokens(
    words: list[WordTimestamp] | None,
    cut: float | None,
    window: float,
    prev_text: str = "",
) -> list[WordTimestamp]:
    """Drop zero-duration overlap echoes on a hard-cut; keep real words like 一 / 五."""
    items = list(words or [])
    if cut is None or window < 0:
        return items
    lo = float(cut) - float(window)
    hi = float(cut) + _HEAD_COLLAPSE_WINDOW
    kept: list[WordTimestamp] = []
    i = 0
    while i < len(items):
        word = items[i]
        mid = (float(word.start) + float(word.end)) / 2.0
        if not (_is_collapsed(word) and lo <= mid <= hi):
            kept.append(word)
            i += 1
            continue
        j = i
        while j < len(items) and _is_collapsed(items[j]):
            mid_j = (float(items[j].start) + float(items[j].end)) / 2.0
            if mid_j > hi:
                break
            j += 1
        run = items[i:j]
        run_text = _join_words(run)
        follow_text = _join_words(items[j : j + max(len(run), 1)])
        if _collapsed_is_overlap(run_text, prev_text, follow_text):
            i = j
            continue
        kept.extend(run)
        i = j
    return kept


def _drop_text_prefix(words: list[WordTimestamp], prefix: str) -> list[WordTimestamp]:
    need = len(prefix or "")
    if need <= 0:
        return list(words)
    acc = 0
    i = 0
    items = list(words)
    while i < len(items) and acc < need:
        acc += len(items[i].word or "")
        i += 1
    return items[i:]


def drop_leading_short_sentence(
    prev_words: list[WordTimestamp] | None,
    words: list[WordTimestamp] | None,
    text: str,
    min_gap: float = _SHORT_GAP_S,
) -> tuple[str, list[WordTimestamp]]:
    """Drop a 1-2 char leftover sentence after a long gap (广告垫字 / 片头)."""
    items = list(words or [])
    raw = text or ""
    match = _LEADING_SHORT.match(raw)
    if not match or not prev_words or not items:
        return raw, items
    gap = float(items[0].start) - float(prev_words[-1].end)
    if gap < min_gap:
        return raw, items
    prefix = match.group(0)
    return raw[len(prefix) :], _drop_text_prefix(items, prefix)


def finalize_offline_chunk(
    asr_text: str,
    words: list[WordTimestamp] | None,
    *,
    audio_start: float,
    audio_end: float,
    drop_before: float | None,
    kind: str,
    prev_text: str = "",
    prev_words: list[WordTimestamp] | None = None,
    eps: float = 0.15,
    overlap: float = 1.5,
) -> tuple[str, list[WordTimestamp]]:
    """Time-drop, collapse-drop, then WS strip_boundary against the previous chunk."""
    shifted = shift_words(words, audio_start)
    kept = drop_overlap_tokens(shifted, drop_before, eps)
    if drop_before is not None and prev_text:
        kept = restore_nonoverlap_dropped(shifted, kept, prev_text)
    if kind == "hard":
        kept = drop_collapsed_cut_tokens(kept, audio_end, overlap, prev_text)
    if drop_before is not None:
        kept = drop_collapsed_cut_tokens(kept, drop_before, _HEAD_COLLAPSE_WINDOW, prev_text)
    if kind == "hard":
        kept = strip_forced_cut_words(kept)

    raw = (asr_text or "").strip()
    text = _join_words(kept) if kept else raw
    if kind == "hard":
        text = strip_forced_cut_stop(text)
        text = trim_dangling_tail(text)
    else:
        text = trim_cut_fragment(text)
    if kept and text != _join_words(kept):
        kept = sync_words_to_text(kept, text)

    if prev_text:
        if leftover_echo(prev_text, text):
            return "", []
        stitched = strip_boundary(prev_text, text, acoustic_overlap=True)
        if stitched != text:
            if kept:
                kept = sync_words_to_text(kept, stitched)
            text = stitched
        if not text:
            return "", []
        text, kept = drop_leading_short_sentence(prev_words, kept, text)
        if leftover_echo(prev_text, text):
            return "", []

    return text or "", list(kept or [])


def append_offline_chunk(
    all_words: list[WordTimestamp],
    chunk_words: list[WordTimestamp],
) -> list[WordTimestamp]:
    """Keep the 1-char / 的地得 word stitch as a safety net."""
    return stitch_chunk_words(all_words, chunk_words)
