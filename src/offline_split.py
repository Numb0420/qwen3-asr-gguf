"""Plan offline ASR chunks: silence-first cuts, hard-cut overlap, timestamp dedup."""
from __future__ import annotations

from dataclasses import dataclass

from backend.base import WordTimestamp


@dataclass(frozen=True)
class SplitConfig:
    target: float = 35.0
    search: float = 5.0
    max_chunk: float = 40.0
    min_silence: float = 0.4
    fallback_silence: float = 0.2
    hard_overlap: float = 1.5
    silence_overlap: float = 0.4
    cut_eps: float = 0.15
    allow_hard_overlap: bool = True


@dataclass(frozen=True)
class ChunkPlan:
    audio_start: float
    audio_end: float
    kind: str  # silence | hard | tail
    drop_before: float | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.audio_end - self.audio_start)


def _silence_mid(region: tuple[float, float]) -> float:
    return (region[0] + region[1]) / 2.0


def _silence_dur(region: tuple[float, float]) -> float:
    return max(0.0, region[1] - region[0])


def _best_silence(
    silences: list[tuple[float, float]],
    *,
    cursor: float,
    lo: float,
    hi: float,
    min_dur: float,
    target: float,
) -> float | None:
    """Return cut time (silence midpoint) or None."""
    best_mid: float | None = None
    best_score = float("inf")
    for region in silences:
        dur = _silence_dur(region)
        if dur < min_dur:
            continue
        # Must overlap the search window.
        overlap_lo = max(region[0], lo)
        overlap_hi = min(region[1], hi)
        if overlap_hi <= overlap_lo:
            continue
        mid = _silence_mid(region)
        if mid <= cursor or mid > hi:
            continue
        score = abs(mid - target) - 0.3 * dur
        if score < best_score:
            best_score = score
            best_mid = mid
    return best_mid


def _find_cut(
    silences: list[tuple[float, float]],
    cursor: float,
    duration: float,
    cfg: SplitConfig,
) -> float | None:
    target = cursor + cfg.target
    max_end = min(duration, cursor + cfg.max_chunk)
    win_lo = max(cursor, target - cfg.search)
    win_hi = min(max_end, target + cfg.search)
    if win_hi <= win_lo:
        return None

    cut = _best_silence(
        silences, cursor=cursor, lo=win_lo, hi=win_hi, min_dur=cfg.min_silence, target=target
    )
    if cut is not None:
        return cut
    cut = _best_silence(
        silences, cursor=cursor, lo=win_lo, hi=win_hi, min_dur=cfg.fallback_silence, target=target
    )
    if cut is not None:
        return cut

    ext_hi = max_end
    if ext_hi > win_hi:
        cut = _best_silence(
            silences, cursor=cursor, lo=win_lo, hi=ext_hi, min_dur=cfg.min_silence, target=target
        )
        if cut is not None:
            return cut
        cut = _best_silence(
            silences, cursor=cursor, lo=win_lo, hi=ext_hi, min_dur=cfg.fallback_silence, target=target
        )
        if cut is not None:
            return cut
    return None


def plan_chunks(
    duration: float,
    silences: list[tuple[float, float]] | None,
    cfg: SplitConfig,
) -> list[ChunkPlan]:
    """Plan [audio_start, audio_end) slices. `cursor` is the owned timeline origin."""
    duration = float(duration)
    if duration <= 0:
        return []
    regions = list(silences or [])
    overlap = cfg.hard_overlap if cfg.allow_hard_overlap else 0.0
    overlap = max(0.0, min(overlap, cfg.max_chunk - 1e-3))

    plans: list[ChunkPlan] = []
    cursor = 0.0
    audio_start = 0.0
    drop_before: float | None = None

    while cursor < duration - 1e-6:
        remaining = duration - cursor
        if remaining <= cfg.max_chunk + 1e-6:
            plans.append(
                ChunkPlan(
                    audio_start=round(audio_start, 3),
                    audio_end=round(duration, 3),
                    kind="tail",
                    drop_before=drop_before,
                )
            )
            break

        cut = _find_cut(regions, cursor, duration, cfg)
        if cut is not None:
            plans.append(
                ChunkPlan(
                    audio_start=round(audio_start, 3),
                    audio_end=round(cut, 3),
                    kind="silence",
                    drop_before=drop_before,
                )
            )
            cursor = cut
            sil_ov = cfg.silence_overlap if cfg.allow_hard_overlap else 0.0
            sil_ov = max(0.0, min(sil_ov, cfg.max_chunk - 1e-3))
            audio_start = max(0.0, cut - sil_ov)
            drop_before = cut if sil_ov > 0 else None
            continue

        cut = min(duration, cursor + cfg.max_chunk)
        plans.append(
            ChunkPlan(
                audio_start=round(audio_start, 3),
                audio_end=round(cut, 3),
                kind="hard",
                drop_before=drop_before,
            )
        )
        drop_before = cut
        cursor = cut
        audio_start = max(0.0, cut - overlap)

    return plans


def shift_words(words: list[WordTimestamp] | None, offset: float) -> list[WordTimestamp]:
    items = list(words or [])
    if not offset:
        return [
            WordTimestamp(word=w.word, start=round(float(w.start), 3), end=round(float(w.end), 3))
            for w in items
        ]
    return [
        WordTimestamp(
            word=w.word,
            start=round(float(w.start) + offset, 3),
            end=round(float(w.end) + offset, 3),
        )
        for w in items
    ]


def drop_overlap_tokens(
    words: list[WordTimestamp] | None,
    cut: float | None,
    eps: float,
) -> list[WordTimestamp]:
    items = list(words or [])
    if cut is None:
        return items
    keep: list[WordTimestamp] = []
    for w in items:
        mid = (float(w.start) + float(w.end)) / 2.0
        if mid >= cut - eps:
            keep.append(w)
    return keep


_SENT_END = set("。！？；!?")
_PUNCT = set("。！？；!?，,、：:;；…—-\"'“”‘’（）()[]《》〈〉 \t")


def _is_punct_word(text: str) -> bool:
    raw = (text or "").strip()
    return not raw or all(ch in _PUNCT for ch in raw)


def _is_sent_end_word(text: str) -> bool:
    return any(ch in _SENT_END for ch in (text or ""))


def stitch_chunk_text(prev: str, nxt: str) -> str:
    """Join chunk transcripts; drop a duplicated boundary char and a false chunk-end period."""
    if not prev:
        return nxt or ""
    if not nxt:
        return prev
    i = len(prev)
    while i > 0 and prev[i - 1] in " \t。！？；!?":
        i -= 1
    if i <= 0:
        return prev + nxt
    k = 0
    while k < len(nxt) and nxt[k] in " \t":
        k += 1
    if k >= len(nxt):
        return prev
    if nxt[k] == prev[i - 1]:
        return prev[:i] + nxt[k + 1 :]
    if nxt[k] in "的地得" and i < len(prev) and prev[i:] and all(ch in " \t。！？；!?" for ch in prev[i:]):
        return prev[:i] + nxt[k:]
    return prev + nxt


def stitch_chunk_words(
    prev: list[WordTimestamp] | None,
    nxt: list[WordTimestamp] | None,
    max_gap: float = 0.5,
) -> list[WordTimestamp]:
    """Merge two chunk token lists and drop a duplicated boundary character."""
    left = list(prev or [])
    right = list(nxt or [])
    if not left or not right:
        return left + right

    i = len(left) - 1
    while i >= 0 and _is_punct_word(left[i].word):
        i -= 1
    j = 0
    while j < len(right) and _is_punct_word(right[j].word):
        j += 1
    if i < 0 or j >= len(right):
        return left + right

    gap = float(right[j].start) - float(left[i].end)
    a = (left[i].word or "").strip().rstrip("。！？；!?")
    b = (right[j].word or "").strip()
    if gap <= max_gap and a and b and a[-1] == b[0]:
        kept_left = [w for k, w in enumerate(left) if not (k > i and _is_sent_end_word(w.word))]
        if len(b) == 1:
            kept_right = right[:j] + right[j + 1 :]
        else:
            trimmed = WordTimestamp(word=b[1:], start=right[j].start, end=right[j].end)
            kept_right = right[:j] + [trimmed] + right[j + 1 :]
        return kept_left + kept_right
    if b[:1] in "的地得" and any(_is_sent_end_word(w.word) for w in left[i + 1 :]):
        kept_left = [w for k, w in enumerate(left) if not (k > i and _is_sent_end_word(w.word))]
        return kept_left + right
    return left + right


def strip_forced_cut_words(words: list[WordTimestamp] | None) -> list[WordTimestamp]:
    """Drop a trailing 。 token after a hard cut."""
    items = list(words or [])
    while items and (items[-1].word or "").strip() == "。":
        items.pop()
    return items
