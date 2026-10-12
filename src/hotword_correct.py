"""Optional phoneme-based hotword correction after ITN, before emit / JSON.

Mirrors `src/itn.py`: takes (text, words), returns corrected (text, words).
Each corrected hotword stays one token and inherits the time range of its
matched original-text span.
"""
from __future__ import annotations

import config
from backend.base import WordTimestamp
from logger import log
from token_sync import sync_words_from_replacements

# Lazy singleton — importing `hotword` pulls in pypinyin + rapidfuzz, so we
# defer it until the first ASR result actually needs correction.
_corrector = None


def _get_corrector():
    global _corrector
    if _corrector is None:
        from hotword import PhonemeCorrector  # noqa: E402  (lazy import)
        if not config.HOTWORDS_PATH.exists():
            log.warning(
                "Hotword init skipped | file not found path={}",
                config.HOTWORDS_PATH,
            )
            _corrector = False  # sentinel: disabled because no file
            return _corrector
        hot_text = config.HOTWORDS_PATH.read_text(encoding="utf-8")
        _corrector = PhonemeCorrector(threshold=config.HOTWORD_THRESHOLD)
        n = _corrector.update_hotwords(hot_text)
        log.info(
            "Hotword corrector loaded | path={} count={} threshold={}",
            config.HOTWORDS_PATH,
            n,
            config.HOTWORD_THRESHOLD,
        )
    return _corrector


def reload_hotwords() -> int:
    """Re-read hotwords.txt and rebuild the index. Returns hotword count.

    Safe to call at runtime after editing hotwords.txt (e.g. via a future
    reload endpoint). Returns -1 if the file is missing.
    """
    global _corrector
    from hotword import PhonemeCorrector  # noqa: E402  (lazy import)
    if not config.HOTWORDS_PATH.exists():
        log.warning("Hotword reload skipped | file not found path={}", config.HOTWORDS_PATH)
        return -1
    hot_text = config.HOTWORDS_PATH.read_text(encoding="utf-8")
    if _corrector is None or _corrector is False:
        _corrector = PhonemeCorrector(threshold=config.HOTWORD_THRESHOLD)
    n = _corrector.update_hotwords(hot_text)
    log.info("Hotword reloaded | path={} count={}", config.HOTWORDS_PATH, n)
    return n


def apply_hotword(
    text: str,
    words: list[WordTimestamp] | None,
) -> tuple[str, list[WordTimestamp]]:
    """Apply phoneme-based hotword correction to `text`.

    Returns (corrected_text, synced_words). If disabled / no file / no change,
    returns the inputs unchanged. On error, logs a warning and returns inputs.
    """
    items = list(words or [])
    raw = text or ""
    if not config.HOTWORD_ENABLED or not raw:
        return raw, items

    corrector = _get_corrector()
    if not corrector:  # None or False sentinel
        return raw, items

    try:
        result = corrector.correct(raw)
    except Exception as exc:
        log.warning("Hotword failed | err={}", exc)
        return raw, items

    if result.text == raw:
        return raw, items

    if result.matches:
        parts = [f"{wrong} -> {right}({score:.2f})" for wrong, right, score in result.matches]
        log.info("Hotword applied | matches={}", " | ".join(parts))

    try:
        replacements = [(m.start, m.end, m.hotword) for m in result.replacements or []]
        cursor = 0
        parts = []
        for start, end, hotword in sorted(replacements):
            if not (cursor <= start < end <= len(raw)):
                raise ValueError("Invalid hotword replacement spans")
            parts.extend([raw[cursor:start], hotword])
            cursor = end
        parts.append(raw[cursor:])
        if "".join(parts) != result.text:
            raise ValueError("Hotword replacement spans do not match corrected text")
        synced = sync_words_from_replacements(items, raw, replacements)
    except Exception as exc:
        log.warning("Hotword timestamp sync failed | err={}", exc)
        return raw, items
    return result.text, synced
