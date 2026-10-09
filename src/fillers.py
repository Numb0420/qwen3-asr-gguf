"""Conservative display cleanup for ASR disfluencies."""
from __future__ import annotations

from difflib import SequenceMatcher
import re


# 呃/嗯 are hesitation sounds even without spaces in Chinese ASR output.
# 哎呀 and 嗯哼 can carry meaning, so leave those combinations intact.
_HESITATION = re.compile(r"呃+(?!逆)|嗯+(?![嗯哼])|哎+(?![哎呀哟])")
# Sentence-initial or comma-separated 啊 is a filler; sentence-final 好啊 is not.
_INITIAL_A = re.compile(r"(?<![\u4e00-\u9fffA-Za-z0-9])啊+")
_DUP_COMMA = re.compile(r"[，,、]\s*[，,、]+")
_DUP_STOP = re.compile(r"[。；;]\s*[。；;]+")
_COMMA_STOP = re.compile(r"[，,、]\s*([。；;！？!?])")
_CJK_SPACE = re.compile(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])")


def clean_fillers(text: str) -> str:
    """Remove common hesitation sounds from displayed ASR text."""
    if not text:
        return text
    cleaned = _HESITATION.sub("", text)
    cleaned = _INITIAL_A.sub("", cleaned)
    if cleaned == text:
        return text
    cleaned = _DUP_COMMA.sub("，", cleaned)
    cleaned = _DUP_STOP.sub("。", cleaned)
    cleaned = _COMMA_STOP.sub(r"\1", cleaned)
    cleaned = _CJK_SPACE.sub("", cleaned)
    cleaned = cleaned.strip()
    if not cleaned.strip("，,、。；;！？!? "):
        return ""
    return cleaned.lstrip("，,、。；;！？!? ").rstrip("，,、 ")


def clean_filler_segments(segments: list[dict]) -> list[dict]:
    """Clean offline text and its aligned chars together, preserving timestamps."""
    output: list[dict] = []
    for segment in segments:
        original = segment["text"]
        cleaned = clean_fillers(original)
        if not cleaned:
            continue
        item = dict(segment)
        item["text"] = cleaned
        item["index"] = len(output) + 1
        if cleaned != original:
            keep: set[int] = set()
            for block in SequenceMatcher(None, original, cleaned, autojunk=False).get_matching_blocks():
                keep.update(range(block.a, block.a + block.size))
            chars: list[dict] = []
            cursor = 0
            for char in segment.get("chars") or []:
                token = char["text"]
                pos = original.find(token, cursor)
                if pos < 0:
                    # Never return timestamps attached to the wrong text.
                    chars = []
                    break
                cursor = pos + len(token)
                kept = "".join(original[i] for i in range(pos, cursor) if i in keep)
                if kept:
                    chars.append({**char, "text": kept})
            item["chars"] = chars
            if chars:
                item["start"] = chars[0]["start"]
                item["end"] = chars[-1]["end"]
            stop = re.search(r"[。！？；!?]+$", cleaned)
            item["punctuation"] = stop.group(0) if stop else ""
        output.append(item)
    return output
