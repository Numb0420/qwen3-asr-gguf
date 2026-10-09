# coding=utf-8
"""Per-window encoder embedding / frozen-text helpers.

encoder_cache[i] is a dict:
  embd       ndarray   full configured window embedding
  text       str       frozen window text (committed when skip_committed_text)
  committed  bool      text is frozen and must not be rewritten
"""
from __future__ import annotations


def is_full_window(slice_samples: int, configured_samples: int) -> bool:
    """True only when the slice is exactly one configured encoder window.

    A shrunk short utterance and a padded tail are not full windows.
    """
    return int(slice_samples) > 0 and int(slice_samples) == int(configured_samples)


def cache_entry(cache, idx):
    if cache is None:
        return None
    return cache.get(idx)


def cache_embd(entry):
    if entry is None:
        return None
    if isinstance(entry, dict):
        return entry.get("embd")
    return entry


def committed_text(entry):
    if not isinstance(entry, dict) or not entry.get("committed"):
        return None
    if "text" not in entry:
        return None
    return entry.get("text") or ""


def raw_window_text(entry):
    """Last decoded window text. Not a freeze flag."""
    if not isinstance(entry, dict):
        return None
    if "raw_text" in entry:
        return entry.get("raw_text") or ""
    if entry.get("text") is not None and not entry.get("committed"):
        return entry.get("text") or ""
    return None


def set_cache_raw_text(cache, idx, text) -> bool:
    """Store raw window text without freezing it."""
    if cache is None:
        return False
    entry = cache.get(idx)
    if not isinstance(entry, dict):
        entry = {} if entry is None else {"embd": entry}
        cache[idx] = entry
    entry["raw_text"] = text if text is not None else ""
    return True


def set_cache_embd(cache, idx, embd):
    if cache is None or embd is None:
        return None
    entry = cache.get(idx)
    if not isinstance(entry, dict):
        entry = {}
        cache[idx] = entry
    entry["embd"] = embd
    return entry


def chunk_commit_ready(
    chunk_idx: int,
    *,
    chunk_size_sec: float,
    total_duration: float,
    margin_sec: float,
    was_last: bool,
    full_window: bool,
) -> bool:
    """Commit only a complete non-last window after the safety margin."""
    if not full_window or was_last:
        return False
    margin = max(0.0, float(margin_sec))
    closed_at = (int(chunk_idx) + 1) * float(chunk_size_sec) + margin
    return float(total_duration) + 1e-9 >= closed_at


def should_skip_decode(cache, idx, *, was_last: bool, enabled: bool = True) -> tuple:
    """Skip decode of a frozen non-last window. Last window always decodes."""
    if not enabled or was_last:
        return False, None
    entry = cache_entry(cache, idx)
    text = committed_text(entry)
    if text is None or cache_embd(entry) is None:
        return False, None
    return True, text


def commit_chunk_text(cache, idx, text) -> bool:
    """Freeze window text. Returns True if this call stored a new commit."""
    if cache is None:
        return False
    entry = cache.get(idx)
    if not isinstance(entry, dict):
        entry = {} if entry is None else {"embd": entry}
        cache[idx] = entry
    if entry.get("committed"):
        return False
    entry["text"] = text if text is not None else ""
    entry["committed"] = True
    return True
