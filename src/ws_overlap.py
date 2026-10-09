"""WebSocket hard-cut overlap, pre-roll, and timeline helpers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.base import WordTimestamp
from offline_split import shift_words

_SENT_END = "。！？；!?"
_JOIN_PUNCT = "，,、 "
_TRAIL_QUOTES = "\"'“”‘’"
_TRAIL_STRIP = " \t" + _SENT_END + _JOIN_PUNCT + _TRAIL_QUOTES


@dataclass(frozen=True)
class HardCutState:
    speech_buf: np.ndarray
    buffer_audio_start: float
    commit_start: float
    drop_before: float
    last_partial_at: int


def realtime_window_sec(duration: float, min_sec: float) -> float:
    """One ASR window for the whole utterance; short clips still pad to min_sec."""
    return max(float(min_sec), float(duration) + 0.05)


def fresh_audio_samples(buffer_size: int, audio_start: float, commit_start: float, sample_rate: int) -> int:
    """Audio newly owned by this utterance, excluding reused left context."""
    context = max(0, int(round((float(commit_start) - float(audio_start)) * int(sample_rate))))
    return max(0, int(buffer_size) - min(int(buffer_size), context))


def append_pre_roll(ring: np.ndarray, chunk: np.ndarray, max_samples: int) -> np.ndarray:
    """Unconditionally keep the latest max_samples of raw PCM."""
    samples = np.asarray(chunk, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return np.asarray(ring, dtype=np.float32).reshape(-1)
    if max_samples <= 0:
        return np.zeros(0, dtype=np.float32)
    base = np.asarray(ring, dtype=np.float32).reshape(-1)
    out = samples if base.size == 0 else np.concatenate([base, samples])
    if out.size > max_samples:
        out = out[-max_samples:]
    return out


def unused_pre_roll(ring: np.ndarray, chunk: np.ndarray) -> np.ndarray:
    """Ring already contains `chunk` at the end; return only samples not in this packet."""
    extra = int(np.asarray(ring).size) - int(np.asarray(chunk).size)
    if extra <= 0:
        return np.zeros(0, dtype=np.float32)
    return np.asarray(ring, dtype=np.float32).reshape(-1)[:extra].copy()


def realtime_buffer_origin(stream_samples: int, buffer_samples: int, sample_rate: int) -> float:
    """Global time of buffer[0] if the buffer is a suffix of the PCM stream."""
    sr = int(sample_rate)
    if sr <= 0:
        return 0.0
    return max(0.0, (int(stream_samples) - int(buffer_samples)) / float(sr))


def discard_stale_overlap_after_skip(skip_samples: int) -> bool:
    """After a silence skip, previous-turn PCM is not next to the new sentence."""
    return int(skip_samples) > 0


def retie_turn_overlap(
    overlap_samples: int,
    following_samples: int,
    stream_samples: int,
    sample_rate: int,
) -> tuple[float, float]:
    """Place reused VAD overlap on the stream tail so a silence gap is not squeezed out.

    Returns `(buffer_audio_start, drop_before)`. `drop_before` is the end of the
    prepended overlap on this new axis; previous-turn tokens stay time-dropped.
    """
    origin = realtime_buffer_origin(
        stream_samples,
        int(overlap_samples) + int(following_samples),
        sample_rate,
    )
    sr = int(sample_rate)
    overlap_s = (int(overlap_samples) / float(sr)) if sr > 0 else 0.0
    return origin, origin + overlap_s


def apply_hard_cut(
    speech_buf: np.ndarray,
    buffer_audio_start: float,
    overlap_sec: float,
    sample_rate: int,
    holdback_sec: float = 0.0,
) -> HardCutState:
    """Keep the last overlap_sec of PCM and roll buffer_audio_start back.

    drop_before / commit_start are the global cut minus holdback, not the last aligner char.
    """
    buf = np.asarray(speech_buf, dtype=np.float32).reshape(-1)
    sr = int(sample_rate)
    overlap_n = max(0, int(round(float(overlap_sec) * sr)))
    duration = float(buf.size) / float(sr) if buf.size and sr else 0.0
    cut_at = float(buffer_audio_start) + duration
    holdback = max(0.0, float(holdback_sec))
    boundary = cut_at - holdback if holdback > 0.0 and duration > holdback + 1.0 else cut_at
    if overlap_n > 0 and buf.size > overlap_n:
        kept = buf[-overlap_n:].copy()
        audio_start = cut_at - (float(kept.size) / float(sr))
    else:
        kept = buf.copy()
        audio_start = float(buffer_audio_start) if overlap_n > 0 else cut_at
        if overlap_n <= 0:
            kept = np.zeros(0, dtype=np.float32)
    return HardCutState(
        speech_buf=kept,
        buffer_audio_start=audio_start,
        commit_start=boundary,
        drop_before=boundary,
        last_partial_at=int(kept.size),
    )


@dataclass(frozen=True)
class SoftCutPlan:
    """Explicit previous/next bounds for a silence-resume soft cut."""
    previous_end: float
    next_start: float
    resume_time: float
    silence_start: float
    overlap_seconds: float


@dataclass(frozen=True)
class SoftCutState:
    """Next-utterance buffer after a split soft cut."""
    speech_buf: np.ndarray
    buffer_audio_start: float
    commit_start: float
    drop_before: float
    last_partial_at: int


def plan_soft_cut(
    buffer_audio_start: float,
    resume_time: float,
    resumed_after_silence_ms: float,
    guard_ms: float,
    buf_end: float,
) -> SoftCutPlan:
    """Place previous_end at mid-silence and next_start before silence_start.

    The current resume packet belongs to the next utterance. Lookback is capped
    by audio already in the buffer, not a blind fixed rewind.
    """
    start = float(buffer_audio_start)
    end = float(buf_end)
    resume = min(max(float(resume_time), start), end if end > start else float(resume_time))
    silence_s = max(0.0, float(resumed_after_silence_ms) / 1000.0)
    silence_start = max(start, resume - silence_s)
    previous_end = (silence_start + resume) / 2.0
    if previous_end >= resume:
        previous_end = max(start, resume - 1e-4)
    previous_end = min(max(previous_end, start), end)
    available = max(0.0, silence_start - start)
    guard_s = min(max(0.0, float(guard_ms) / 1000.0), available)
    next_start = max(start, silence_start - guard_s)
    if next_start >= previous_end:
        overlap_s = max(guard_s, (resume - silence_start) / 2.0, 0.02)
        next_start = max(start, previous_end - overlap_s)
    if next_start > previous_end:
        next_start = previous_end
    return SoftCutPlan(
        previous_end=previous_end,
        next_start=next_start,
        resume_time=resume,
        silence_start=silence_start,
        overlap_seconds=max(0.0, previous_end - next_start),
    )


def split_soft_cut(
    speech_buf: np.ndarray,
    buffer_audio_start: float,
    plan: SoftCutPlan,
    sample_rate: int,
) -> tuple[np.ndarray, SoftCutState]:
    """Split PCM into previous submit audio and the next utterance buffer."""
    sr = int(sample_rate) if int(sample_rate) > 0 else 1
    buf = np.asarray(speech_buf, dtype=np.float32).reshape(-1)
    origin = float(buffer_audio_start)

    def _offset(t: float) -> int:
        n = int(round((float(t) - origin) * float(sr)))
        return max(0, min(n, buf.size))

    prev_off = _offset(plan.previous_end)
    next_off = _offset(plan.next_start)
    previous_audio = buf[:prev_off].copy()
    kept = buf[next_off:].copy() if next_off < buf.size else np.zeros(0, dtype=np.float32)
    return previous_audio, SoftCutState(
        speech_buf=kept,
        buffer_audio_start=plan.next_start,
        commit_start=plan.previous_end,
        drop_before=plan.previous_end,
        last_partial_at=0,
    )


def apply_soft_cut(
    speech_buf: np.ndarray,
    buffer_audio_start: float,
    cut_point: float,
    sample_rate: int,
    *,
    resumed_after_silence_ms: float = 0.0,
    guard_ms: float = 0.0,
    buf_end: float | None = None,
) -> SoftCutState:
    """Keep the next-utterance slice. Prefer split_soft_cut for previous audio."""
    duration = float(np.asarray(speech_buf).size) / float(sample_rate) if sample_rate else 0.0
    end = float(buf_end) if buf_end is not None else float(buffer_audio_start) + duration
    plan = plan_soft_cut(
        buffer_audio_start,
        cut_point,
        resumed_after_silence_ms,
        guard_ms,
        end,
    )
    _prev, state = split_soft_cut(speech_buf, buffer_audio_start, plan, sample_rate)
    return state


def fallback_empty_final(
    raw_text: str,
    finalized: str,
    last_partial: str,
    *,
    overlap_prefix: str = "",
    drop_before: float | None = None,
) -> str:
    """If the model returned nothing, reuse the last non-empty partial for this utterance."""
    if finalized:
        return finalized
    if raw_text:
        return finalized
    prev = last_partial or ""
    if not prev:
        return ""
    if drop_before is not None:
        return strip_boundary(overlap_prefix, prev, acoustic_overlap=True)
    return prev


def trim_cut_fragment(text: str, max_chars: int = 2) -> str:
    """Drop a 1-2 char leftover after a real sentence end: '决议公布。决。' → '决议公布。'"""
    s = (text or "").rstrip()
    if not s:
        return s
    if s[-1] not in _SENT_END:
        ch = s[-1]
        if len(s) >= 2 and s[-2] in _SENT_END and "\u4e00" <= ch <= "\u9fff":
            return s[:-1]
        return s
    j = len(s) - 1
    while j > 0 and s[j] in " \t" + _SENT_END:
        j -= 1
    i = j
    while i > 0 and s[i] not in _SENT_END:
        i -= 1
    if i <= 0 or s[i] not in _SENT_END:
        return s
    frag = s[i + 1 : j + 1].strip()
    if 1 <= len(frag) <= max_chars:
        return s[: i + 1]
    return s


def trim_dangling_tail(text: str) -> str:
    """Drop a cut-off char after 的/地/得: '武装夺取政权的政。' → '武装夺取政权的'."""
    s = trim_cut_fragment(text)
    if len(s) >= 3 and s[-1] in _SENT_END and s[-3] in "的地得":
        ch = s[-2]
        if "\u4e00" <= ch <= "\u9fff":
            return s[:-2]
    return s


def stitch_prev_text(text: str) -> str:
    """Previous-final lookup key. Do not rewrite characters."""
    return (text or "").rstrip()


def _drop_plain_suffix(text: str, n: int) -> str:
    """Drop the last n non-punctuation characters, including punct after them."""
    if n <= 0:
        return text or ""
    count = 0
    i = len(text)
    while i > 0 and count < n:
        i -= 1
        if text[i] not in _PUNCT_CHARS:
            count += 1
    return text[:i]


@dataclass(frozen=True)
class PartialWindowPlan:
    offset: int
    keep: int
    tail: bool
    head: bool
    step: bool = False
    complete: bool = False


def plan_partial_window(
    n_samples: int,
    cap: int,
    *,
    has_shown: bool,
    sample_rate: int = 16000,
    origin_start: float = 0.0,
    prev_window_end: float | None = None,
    block_start: float | None = None,
) -> PartialWindowPlan | None:
    """Re-decode the current open block; completed blocks never re-enter it.

    Repeated partials of one block replace each other. At cap the block is
    committed, and the next block starts at its exact end (no PCM overlap).
    """
    n = int(n_samples)
    c = int(cap)
    if c <= 0 or n <= 0:
        return None
    sr = max(1, int(sample_rate))
    if prev_window_end is None and has_shown:
        return None
    if prev_window_end is not None and origin_start + n / sr <= prev_window_end + 0.5 / sr:
        return PartialWindowPlan(offset=n, keep=0, tail=True, head=False)
    anchor = origin_start if block_start is None else float(block_start)
    start_offset = int(round((anchor - float(origin_start)) * sr))
    start_offset = max(0, min(start_offset, n))
    remaining = n - start_offset
    if remaining <= 0:
        return PartialWindowPlan(offset=n, keep=0, tail=True, head=False, step=False)
    keep = min(remaining, c)
    return PartialWindowPlan(
        offset=start_offset,
        keep=keep,
        tail=True,
        head=start_offset == 0,
        step=remaining > c,
        complete=remaining >= c,
    )


def pending_caption_prefix(
    shown: dict[int, str],
    finalized: set[int],
    current_utt: int,
) -> str:
    """Text still on screen from earlier utterances whose final has not been sent.

    Replace-semantics clients show only the latest partial. A new utterance's
    first window would otherwise wipe those words until the previous final
    arrives several seconds later.
    """
    parts: list[str] = []
    for uid in sorted(shown):
        if uid >= int(current_utt) or uid in finalized:
            continue
        piece = shown.get(uid) or ""
        if piece:
            parts.append(piece)
    return "".join(parts)


def join_partial_tail(prev: str, tail: str) -> str:
    """Append one non-overlapping partial window without text rewriting."""
    return (prev or "") + (tail or "")


def strip_partial_terminal_punct(text: str) -> str:
    """Hide sentence punctuation invented at a partial PCM window boundary.

    The complete utterance is decoded again for final, so this only affects
    the provisional display. Internal punctuation and lexical text stay put.
    """
    s = (text or "").rstrip()
    quotes = s[len(s.rstrip(_TRAIL_QUOTES)):]
    body = s[:len(s) - len(quotes)] if quotes else s
    return body.rstrip(_SENT_END) + quotes


class PartialRequestGate:
    """Allow one partial at a time per utterance; remember a newer request."""

    def __init__(self) -> None:
        self.running: set[int] = set()
        self.pending: set[int] = set()

    def defer_if_running(self, utt_id: int) -> bool:
        if utt_id not in self.running:
            return False
        self.pending.add(utt_id)
        return True

    def started(self, utt_id: int, *, more_audio: bool = False) -> None:
        self.running.add(utt_id)
        if more_audio:
            self.pending.add(utt_id)

    def finished(self, utt_id: int, current_utt_id: int) -> bool:
        self.running.discard(utt_id)
        wanted = utt_id in self.pending
        self.pending.discard(utt_id)
        return wanted and utt_id == current_utt_id

    def reset(self) -> None:
        self.running.clear()
        self.pending.clear()


class OrderedFinalBuffer:
    """Release completed finals in audio order, not inference completion order."""

    def __init__(self, next_utt_id: int = 0) -> None:
        self.next_utt_id = int(next_utt_id)
        self.pending: dict[int, dict | None] = {}

    def put(self, utt_id: int, payload: dict | None) -> list[tuple[int, dict | None]]:
        uid = int(utt_id)
        if uid < self.next_utt_id:
            return []
        self.pending[uid] = payload
        ready: list[tuple[int, dict | None]] = []
        while self.next_utt_id in self.pending:
            current = self.next_utt_id
            ready.append((current, self.pending.pop(current)))
            self.next_utt_id += 1
        return ready

    def reset(self, next_utt_id: int) -> None:
        self.next_utt_id = int(next_utt_id)
        self.pending.clear()


def strip_forced_cut_stop(text: str) -> str:
    """A 20s hard cut is not a sentence end; drop the decoder's trailing 。 / quotes."""
    s = (text or "").rstrip().rstrip(_TRAIL_QUOTES).rstrip()
    if s.endswith("。"):
        return s[:-1].rstrip(_TRAIL_QUOTES).rstrip()
    return s


def strip_boundary(prev: str, text: str, *, acoustic_overlap: bool = False) -> str:
    """Stitch the next utterance head against the previous final.

    Prefer an exact suffix-prefix of 2+ chars (including punct).
    1-char mid-phrase strip is allowed only when ``acoustic_overlap`` is true.
    True sentence ends always require 2+ chars. Uncertain matches are kept.
    """
    if not text:
        return ""
    if not prev:
        return text
    raw = strip_overlap_prefix(prev, text, acoustic_overlap=acoustic_overlap)
    strong = 0
    for n in range(min(len(prev), len(text)), 1, -1):
        if text.startswith(prev[-n:]):
            strong = n
            break
    if strong >= 2:
        out = _lstrip_overlap(text[strong:], text)
        return _drop_punct_only(out)
    return _drop_punct_only(raw)


def sync_words_to_text(words: list[WordTimestamp], text: str) -> list[WordTimestamp]:
    """Keep the word span that matches text after a head/tail trim."""
    items = list(words or [])
    raw = "".join(w.word for w in items)
    if not items or raw == text:
        return items
    if text and raw.endswith(text):
        drop_n = len(raw) - len(text)
        acc = 0
        i = 0
        while i < len(items) and acc < drop_n:
            acc += len(items[i].word)
            i += 1
        return items[i:]
    if text and raw.startswith(text):
        keep_n = len(text)
        acc = 0
        i = 0
        while i < len(items) and acc < keep_n:
            acc += len(items[i].word)
            i += 1
        return items[:i]
    return items


def _overlap_floor(prev: str, acoustic_overlap: bool = False) -> int:
    """Minimum suffix-prefix length that may be stripped.

    Sentence-end punctuation always needs 2+ chars (「实现。」+「现在」).
    Mid-phrase 1-char strip is allowed only with an acoustic overlap marker.
    Without that marker, keep the duplicate rather than guessing.
    """
    s = (prev or "").rstrip().rstrip(_TRAIL_QUOTES).rstrip()
    if s and s[-1] in _SENT_END:
        return 2
    return 1 if acoustic_overlap else 2


def strip_overlap_prefix(prev: str, text: str, *, acoustic_overlap: bool = False) -> str:
    """Drop a suffix-prefix overlap so partials do not need ForcedAligner."""
    if not text:
        return ""
    if not prev:
        return text
    floor = _overlap_floor(prev, acoustic_overlap=acoustic_overlap)
    # Layer 1: punctuation-blind suffix-prefix. 这里一定有。” / 这里一定有两个…
    prev_plain = _plain(prev)
    body_plain = _plain(text)
    max_n = min(len(prev_plain), len(body_plain))
    for n in range(max_n, floor - 1, -1):
        if body_plain.startswith(prev_plain[-n:]):
            out = _drop_plain_head(text, n)
            return _drop_punct_only(_lstrip_overlap(out, text))
    # Layer 2: hard-cut leftover + overlap (打牺牲… / 制，确立…).
    i = len(prev)
    while i > 0 and prev[i - 1] in _TRAIL_STRIP:
        i -= 1
    if i <= 0:
        return text
    k = 0
    while k < len(text) and text[k] in " \t":
        k += 1
    body = text[k:]
    core = prev[:i]
    max_n = min(len(core), len(body), 12)
    for n in range(max_n, floor - 1, -1):
        needle = core[-n:]
        idx = body.find(needle)
        if idx < 0 or idx > 2:
            continue
        if idx != 0:
            junk = body[:idx]
            # n>=4: 打 + 牺牲积累的一系列. n<4: 制，确立… but not 刷着手机 / 刷手机.
            if n < 2 or not _is_cut_junk(junk):
                continue
            if n < 4 and _is_phrase_restart(core, junk, needle):
                continue
        out = _lstrip_join(body[idx + n :], text)
        return _drop_punct_only(out)
    return text


def _lstrip_join(out: str, original: str) -> str:
    if out == original:
        return out
    return out.lstrip(_JOIN_PUNCT)


def _lstrip_overlap(out: str, original: str) -> str:
    if out == original:
        return out
    return out.lstrip(_TRAIL_STRIP)


def _drop_plain_head(text: str, n: int) -> str:
    """Drop the first n non-punctuation characters, keeping later punct."""
    count = 0
    i = 0
    while i < len(text) and count < n:
        if text[i] not in _PUNCT_CHARS:
            count += 1
        i += 1
    return text[i:]


def _is_cut_junk(s: str) -> bool:
    """1 CJK, optionally with punct — a hard-cut leftover, not a real word."""
    return len(_plain(s)) <= 1


def _is_phrase_restart(prev: str, junk: str, needle: str) -> bool:
    """True when junk+needle restarts a phrase already at the prev tail.

    刷着手机 + 刷手机: 刷 is the start of the same phrase, not a cut leftover.
    统一、外交 + 一，外交: 一 is the split tail of 统一, still a real overlap.
    """
    prev_plain = _plain(prev)
    junk_plain = _plain(junk)
    if not junk_plain or not needle or not prev_plain.endswith(needle):
        return False
    prefix = prev_plain[: -len(needle)]
    for gap in (1, 2):
        span = len(junk_plain) + gap
        if len(prefix) >= span and prefix[-span:-gap] == junk_plain:
            return True
    return False


def _drop_punct_only(text: str) -> str:
    return "" if text and not _plain(text) else text


_SHORT_UTT_CJK = 4
_SHORT_CLAUSE_CJK = 4


def _cjk_count(text: str) -> int:
    return sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")


def _last_clause(text: str) -> str:
    seps = set("。！？；!?，,、：:")
    for i in range(len(text) - 1, -1, -1):
        if text[i] in seps:
            return text[i + 1 :].strip()
    return text.strip()


def text_ends_sentence(text: str) -> bool:
    """True if VAD may emit a final now.

    ？！ are trusted. Trailing 。 is not: Qwen partials add it on unfinished
    speech. Short whole replies (好的。/知道了。) still count as done so
    meetings are not delayed 1.2s. Only used as a cut gate — does not edit text.
    """
    s = (text or "").rstrip().rstrip(_TRAIL_QUOTES).rstrip()
    if not s or s[-1] not in _SENT_END:
        return False
    # ？！ are rarely faked on a half phrase.
    if s[-1] in "！？!?":
        return True
    # 好的。 / 知道了。 / 没问题。 — whole turn is already a reply.
    if _cjk_count(s) <= _SHORT_UTT_CJK:
        return True
    body = s.rstrip("。；;").rstrip(_TRAIL_QUOTES).rstrip()
    if not body:
        return False
    clause = _last_clause(body)
    n = _cjk_count(clause)
    # ，蓝色海洋。 / 、滨海等。 / ，确。 — last clause is too short to trust.
    if n <= _SHORT_CLAUSE_CJK:
        return False
    return True


_CLAUSE_SEPS = "，,、：:；;"


def leftover_echo(prev: str, text: str) -> bool:
    """True if this utterance is only a short repeat of the previous tail."""
    t = _plain(text)
    if not t:
        return True
    if not prev or len(t) > 4:
        return False
    p = _plain(prev)
    return bool(p) and (p.endswith(t) or t in p[-(len(t) + 2) :])


_PUNCT_CHARS = set("。！？；!?，,、：:;；…—-\"'“”‘’（）()[]《》〈〉 \t")


def _plain(text: str) -> str:
    return "".join(ch for ch in (text or "") if ch not in _PUNCT_CHARS)


def plain_suffix_prefix_n(prev: str, text: str) -> int:
    """Longest punctuation-blind suffix-prefix overlap length."""
    prev_plain = _plain(prev)
    body_plain = _plain(text)
    max_n = min(len(prev_plain), len(body_plain))
    for n in range(max_n, 0, -1):
        if body_plain.startswith(prev_plain[-n:]):
            return n
    return 0


def _plain_head_words(words: list[WordTimestamp], n: int) -> list[WordTimestamp]:
    """Word tokens covering the first n non-punctuation characters."""
    if n <= 0 or not words:
        return []
    out: list[WordTimestamp] = []
    count = 0
    for w in words:
        token = w.word or ""
        plain = sum(1 for ch in token if ch not in _PUNCT_CHARS)
        if plain <= 0:
            continue
        out.append(w)
        count += plain
        if count >= n:
            break
    return out


def _drop_plain_head_words(words: list[WordTimestamp], n: int) -> list[WordTimestamp]:
    head = _plain_head_words(words, n)
    if not head:
        return list(words or [])
    skip = set(id(w) for w in head)
    return [w for w in words if id(w) not in skip]


def overlap_touches_old(
    words: list[WordTimestamp] | None,
    audio_start: float,
    drop_before: float | None,
    plain_n: int,
) -> bool:
    """True if any of the first plain_n chars contacts the previous audio interval."""
    if drop_before is None or plain_n <= 0 or not words:
        return False
    shifted = shift_words(words, audio_start)
    cut = float(drop_before)
    for w in _plain_head_words(shifted, plain_n):
        start = float(w.start)
        end = float(w.end)
        if start < cut or start <= cut <= end:
            return True
    return False


@dataclass(frozen=True)
class OverlapDecision:
    """How a WS final overlap was resolved. match_n is punctuation-blind."""
    match_n: int = 0
    match_plain: str = ""
    acoustic_overlap: bool = False
    contact_old: bool | None = None
    removed_plain: str = ""


def dropped_is_overlap(dropped_text: str, prev: str) -> bool:
    """True if time-dropped chars are a tail of the previous final (real overlap)."""
    dropped = _plain(dropped_text)
    core = _plain(stitch_prev_text(prev))
    if not dropped:
        return True
    if not core:
        return False
    tail = core[-(len(dropped) + 4) :]
    if dropped in tail:
        return True
    max_n = min(len(core), len(dropped))
    for n in range(max_n, 0, -1):
        if dropped.startswith(core[-n:]) or core.endswith(dropped[:n]):
            return True
    return False


def restore_nonoverlap_dropped(
    shifted: list[WordTimestamp],
    kept: list[WordTimestamp],
    prev_text: str,
) -> list[WordTimestamp]:
    """Keep time-dropped chars that are new content sitting in the overlap window.

    Only restore 1-2 chars (sentence-initial 中/和). Three or more dropped
    chars are the previous final's tail, even if `prev_text` is still stale.
    """
    if not shifted or not prev_text:
        return kept
    kept_keys = {(w.word, w.start, w.end) for w in kept}
    dropped = [w for w in shifted if (w.word, w.start, w.end) not in kept_keys]
    if not dropped:
        return kept
    dropped_text = "".join(w.word for w in dropped)
    if len(_plain(dropped_text)) > 2:
        return kept
    if dropped_is_overlap(dropped_text, prev_text):
        return kept
    return dropped + kept


def finalize_ws_utterance(
    words: list[WordTimestamp] | None,
    text: str,
    *,
    audio_start: float,
    audio_duration: float,
    commit_start: float,
    drop_before: float | None,
    eps: float,
    overlap_prefix: str = "",
) -> tuple[str, float, float, list[WordTimestamp], OverlapDecision]:
    """Dedupe a WS final: text overlap first, timestamps only confirm.

    n is the punctuation-blind suffix-prefix length on unmodified raw text.
    Never time-drop first and never guess an extra character. 1-char matches stay.
    """
    del eps  # kept in the signature; cut confirmation does not use the offline eps.
    audio_end = float(audio_start) + float(audio_duration)
    raw = text or ""
    shifted = shift_words(words, audio_start)
    acoustic = bool(drop_before is not None or float(audio_start) < float(commit_start))
    match_n = plain_suffix_prefix_n(overlap_prefix, raw) if overlap_prefix else 0
    match_plain = _plain(raw)[:match_n] if match_n else ""
    contact: bool | None = None
    remove_n = 0
    if match_n >= 2 and acoustic:
        if shifted and drop_before is not None:
            contact = overlap_touches_old(words, audio_start, drop_before, match_n)
            if contact:
                remove_n = match_n
        else:
            contact = None
            remove_n = match_n
    out_text = raw
    kept = list(shifted)
    if remove_n:
        out_text = _drop_punct_only(_lstrip_overlap(_drop_plain_head(raw, remove_n), raw))
        kept = _drop_plain_head_words(shifted, remove_n)
        if out_text:
            kept = sync_words_to_text(kept, out_text)
        else:
            kept = []
    decision = OverlapDecision(
        match_n=match_n,
        match_plain=match_plain,
        acoustic_overlap=acoustic,
        contact_old=contact,
        removed_plain=_plain(raw)[:remove_n] if remove_n else "",
    )
    if kept and out_text:
        start = max(float(commit_start), float(kept[0].start))
        end = float(kept[-1].end)
        return out_text, round(start, 2), round(end, 2), kept, decision
    start = float(commit_start)
    end = max(start, audio_end)
    if out_text:
        return out_text, round(start, 2), round(end, 2), [], decision
    return "", round(start, 2), round(end, 2), [], decision
