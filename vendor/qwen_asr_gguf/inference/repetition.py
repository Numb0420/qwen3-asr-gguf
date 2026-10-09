"""Repetition guards for Qwen3-ASR (official collapse + decode loop abort).

Official parse_asr_output uses character/pattern collapse with a high threshold so
spoken 对对对 / 没有没有没有 is kept. Decode-time abort uses a stricter consecutive
phrase detector and only trims the looping tail.
"""
from __future__ import annotations


def collapse_repetitions(text: str, threshold: int = 10) -> str:
    """Port of Qwen3-ASR utils.detect_and_fix_repetitions (threshold 10, not 3)."""
    if not text:
        return text
    text = _fix_char_repeats(text, threshold)
    return _fix_pattern_repeats(text, threshold)


def _fix_char_repeats(s: str, thresh: int) -> str:
    res: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        count = 1
        while i + count < n and s[i + count] == s[i]:
            count += 1
        if count > thresh:
            res.append(s[i])
            i += count
        else:
            res.append(s[i : i + count])
            i += count
    return "".join(res)


def _fix_pattern_repeats(s: str, thresh: int, max_len: int = 20) -> str:
    n = len(s)
    min_repeat_chars = thresh * 2
    if n < min_repeat_chars:
        return s

    i = 0
    result: list[str] = []
    found = False
    while i <= n - min_repeat_chars:
        found = False
        for k in range(1, max_len + 1):
            if i + k * thresh > n:
                break
            pattern = s[i : i + k]
            valid = True
            for rep in range(1, thresh):
                start_idx = i + rep * k
                if s[start_idx : start_idx + k] != pattern:
                    valid = False
                    break
            if valid:
                end_index = i + thresh * k
                while end_index + k <= n and s[end_index : end_index + k] == pattern:
                    end_index += k
                result.append(pattern)
                result.append(_fix_pattern_repeats(s[end_index:], thresh, max_len))
                i = n
                found = True
                break
        if found:
            break
        result.append(s[i])
        i += 1
    if not found:
        result.append(s[i:])
    return "".join(result)


def suffix_loop_pattern(
    text: str,
    min_reps: int = 6,
    max_k: int = 16,
    min_char_reps: int = 12,
) -> str | None:
    """If `text` ends with consecutive copies of a short phrase, return that phrase.

    Single-char loops (e.g. "对对对…") require ``min_char_reps`` (default 12)
    repetitions to avoid aborting on natural spoken repetitions in meetings.
    Multi-char phrase loops keep ``min_reps`` (default 6).
    """
    n = len(text)
    for k in range(1, max_k + 1):
        required_reps = min_char_reps if k == 1 else min_reps
        need = k * required_reps
        if n < need:
            continue
        pat = text[n - k : n]
        if not pat.strip():
            continue
        if all(text[n - (r + 1) * k : n - r * k] == pat for r in range(required_reps)):
            return pat
    return None


def trim_loop_tail(text: str, pattern: str, keep: int = 2) -> str:
    """Drop extra trailing copies of `pattern`, keep `keep` occurrences."""
    if not text or not pattern:
        return text
    reps = 0
    t = text
    plen = len(pattern)
    while len(t) >= plen and t.endswith(pattern):
        t = t[: -plen]
        reps += 1
    if reps < keep:
        return text
    return t + pattern * keep


def max_new_tokens_for_audio(duration_sec: float, per_sec: float = 8.0, cap: int = 512, floor: int = 32) -> int:
    if duration_sec <= 0:
        return floor
    return int(min(cap, max(floor, duration_sec * per_sec)))
