"""Optional Chinese number ITN after ASR stitch, before emit / JSON."""
from __future__ import annotations

import config
from backend.base import WordTimestamp
from chinese_itn import chinese_to_num
from logger import log
from token_sync import sync_words_from_text


def apply_chinese_itn(
    text: str,
    words: list[WordTimestamp] | None,
) -> tuple[str, list[WordTimestamp]]:
    items = list(words or [])
    raw = text or ""
    if not config.FORMAT_NUM or not raw:
        return raw, items
    try:
        converted = chinese_to_num(raw)
    except Exception as exc:
        log.warning("ITN failed | err={}", exc)
        return raw, items
    if converted == raw:
        return raw, items
    return converted, sync_words_from_text(items, converted)
