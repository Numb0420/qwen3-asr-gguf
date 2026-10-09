"""No-GPU checks for WS VAD timing, hard-cut prefix, and queue coalesce.

    conda activate lingting
    python E2Etest/test_ws_tune.py
"""
from __future__ import annotations

import asyncio
import importlib.util
import inspect
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
VENDOR = ROOT / "vendor"
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import numpy as np

from backend.base import WordTimestamp
from utterance_cache import UtteranceEncoderCache, full_window_count

_chunk_spec = importlib.util.spec_from_file_location(
    "chunk_cache",
    VENDOR / "qwen_asr_gguf" / "inference" / "chunk_cache.py",
)
_chunk_cache = importlib.util.module_from_spec(_chunk_spec)
_chunk_spec.loader.exec_module(_chunk_cache)
chunk_commit_ready = _chunk_cache.chunk_commit_ready
commit_chunk_text = _chunk_cache.commit_chunk_text
is_full_window = _chunk_cache.is_full_window
set_cache_embd = _chunk_cache.set_cache_embd
set_cache_raw_text = _chunk_cache.set_cache_raw_text
should_skip_decode = _chunk_cache.should_skip_decode
raw_window_text = _chunk_cache.raw_window_text
from inference_queue import PriorityInferQueue
from repetition import carry_prefix_tail
from vad import (
    RealtimeVad,
    advance_energy_run,
    choose_hard_cut_silence,
    energy_onset_ready,
    silence_threshold_samples,
)
from ws_overlap import (
    OrderedFinalBuffer,
    PartialRequestGate,
    append_pre_roll,
    apply_hard_cut,
    fallback_empty_final,
    fresh_audio_samples,
    finalize_ws_utterance,
    leftover_echo,
    discard_stale_overlap_after_skip,
    join_partial_tail,
    pending_caption_prefix,
    plan_partial_window,
    plan_soft_cut,
    realtime_buffer_origin,
    realtime_window_sec,
    restore_nonoverlap_dropped,
    retie_turn_overlap,
    split_soft_cut,
    stitch_prev_text,
    strip_partial_terminal_punct,
    strip_boundary,
    strip_forced_cut_stop,
    strip_overlap_prefix,
    text_ends_sentence,
    trim_cut_fragment,
    trim_dangling_tail,
    unused_pre_roll,
    dropped_is_overlap,
)


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_carry_prefix_tail() -> None:
    _assert(carry_prefix_tail("前文然后发文章？对。") == "文章？对。", carry_prefix_tail("前文然后发文章？对。"))
    _assert(carry_prefix_tail("one two three four five six") == "two three four five six", carry_prefix_tail("one two three four five six"))
    _assert(carry_prefix_tail("") == "", "empty")
    _assert(carry_prefix_tail("你好") == "你好", carry_prefix_tail("你好"))


def test_two_tier_silence() -> None:
    sr = 16000
    early = silence_threshold_samples(int(0.2 * sr), sr, flush_ms=600, early_ms=1000, committed_s=0.8)
    committed = silence_threshold_samples(int(0.9 * sr), sr, flush_ms=600, early_ms=1000, committed_s=0.8)
    _assert(early == int(sr * 1.0), str(early))
    _assert(committed == int(sr * 0.6), str(committed))


def test_no_rms_fallback() -> None:
    src = inspect.getsource(RealtimeVad.accept_audio)
    _assert("0.01" not in src, "RMS fallback should be removed")
    _assert("sqrt" not in src, src)
    drain_at = src.find("_drain_queued_segments")
    wave_at = src.find("accept_waveform")
    _assert(0 <= drain_at < wave_at, "drain sherpa segments before accept_waveform")


def test_energy_run_resets_on_silence() -> None:
    sr = 16000
    loud = np.full(int(0.1 * sr), 0.05, dtype=np.float32)
    quiet = np.zeros(int(0.1 * sr), dtype=np.float32)
    run = advance_energy_run(0, loud)
    _assert(run == loud.size, str(run))
    _assert(not energy_onset_ready(run, sr), "100ms is under 200ms")
    run = advance_energy_run(run, loud)
    _assert(energy_onset_ready(run, sr), "200ms of energy should onset")
    run = advance_energy_run(run, quiet)
    _assert(run == 0, "silence resets the run")
    _assert(not energy_onset_ready(run, sr), "reset run must not onset")


def test_drain_pops_queued_segments() -> None:
    class _FakeVad:
        def __init__(self) -> None:
            self.left = 2
            self.pops = 0

        def empty(self) -> bool:
            return self.left <= 0

        def pop(self) -> None:
            self.left -= 1
            self.pops += 1

    vad = RealtimeVad.__new__(RealtimeVad)
    vad._vad = _FakeVad()
    vad._drain_queued_segments()
    _assert(vad._vad.pops == 2, str(vad._vad.pops))
    vad._drain_queued_segments()
    _assert(vad._vad.pops == 2, "empty queue stays empty")


def test_skip_keeps_sustained_energy() -> None:
    src = (SRC / "server.py").read_text(encoding="utf-8")
    _assert("advance_energy_run" in src, "skip path must track energy")
    _assert("energy_open" in src, "sustained energy must open the buffer")
    _assert("WS energy_onset" in src, "energy onset must be logged")
    _assert("WS energy_end" in src, "energy silence must end the utterance")


async def _test_final_drops_pending_partial() -> None:
    q = PriorityInferQueue()
    q.start()

    def job(name: str, delay: float):
        def _fn():
            time.sleep(delay)
            return name
        return _fn

    try:
        t1 = asyncio.create_task(q.submit(job("p1", 0.15), priority=0, kind="partial", session_id="s1", utt_id=1))
        await asyncio.sleep(0.03)
        t2 = asyncio.create_task(q.submit(job("p2", 0.0), priority=0, kind="partial", session_id="s1", utt_id=1))
        t3 = asyncio.create_task(q.submit(job("f", 0.0), priority=0, kind="final", session_id="s1", utt_id=1))
        r1, r2, r3 = await asyncio.gather(t1, t2, t3)
        _assert(r3 == "f", f"final should run, got {r3!r}")
        _assert(r2 is None, f"pending partial should be dropped, got {r2!r}")
        _assert(r1 in ("p1", None), f"in-flight partial may finish, got {r1!r}")
    finally:
        q.stop()


def test_queue_final_drops_partial() -> None:
    asyncio.run(_test_final_drops_pending_partial())


def test_hard_cut_rolls_audio_start() -> None:
    sr = 16000
    buf = np.ones(int(20 * sr), dtype=np.float32)
    state = apply_hard_cut(buf, 0.0, 1.0, sr)
    _assert(state.speech_buf.size == sr, str(state.speech_buf.size))
    _assert(abs(state.buffer_audio_start - 19.0) < 1e-6, str(state.buffer_audio_start))
    _assert(abs(state.commit_start - 20.0) < 1e-6, str(state.commit_start))
    _assert(abs(state.drop_before - 20.0) < 1e-6, str(state.drop_before))
    _assert(state.last_partial_at == sr, str(state.last_partial_at))
    _assert(state.commit_start == state.drop_before, "drop_before must be the commit cut")


def test_hard_cut_silence_selection() -> None:
    regions = [(2.0, 2.4), (8.9, 9.2), (9.5, 9.7), (10.0, 10.5)]
    chosen = choose_hard_cut_silence(regions, 10.0, 5.0, 300)
    _assert(chosen == (8.9, 9.2), f"choose nearest qualifying silence before target, got {chosen}")
    _assert(
        choose_hard_cut_silence([(9.5, 9.7)], 10.0, 5.0, 300) is None,
        "ignore short VAD gaps",
    )
    _assert(
        choose_hard_cut_silence([(4.0, 4.4)], 10.0, 5.0, 300) is None,
        "ignore silences outside the search window",
    )


def test_drop_before_is_commit_not_last_char() -> None:
    words = [
        WordTimestamp("旧", 0.50, 0.72),
        WordTimestamp("新", 1.10, 1.30),
    ]
    text, start, end, kept, _dec = finalize_ws_utterance(
        words,
        "旧新",
        audio_start=19.0,
        audio_duration=4.0,
        commit_start=20.0,
        drop_before=20.0,
        eps=0.15,
    )
    _assert([w.word for w in kept] == ["旧", "新"], kept)
    _assert(start >= 20.0, str(start))
    leaked = finalize_ws_utterance(
        words,
        "旧新",
        audio_start=19.0,
        audio_duration=4.0,
        commit_start=20.0,
        drop_before=19.72,
        eps=0.15,
    )[3]
    _assert([w.word for w in leaked] == ["旧", "新"], "time-only 1-char drop is not used")


def test_strip_overlap_prefix() -> None:
    _assert(strip_overlap_prefix("文章？对。", "对。今天继续") == "对。今天继续", strip_overlap_prefix("文章？对。", "对。今天继续"))
    _assert(strip_overlap_prefix("实现。", "现国家") == "现国家", strip_overlap_prefix("实现。", "现国家"))
    _assert(strip_overlap_prefix("实现。", "现在进入新时代") == "现在进入新时代", strip_overlap_prefix("实现。", "现在进入新时代"))
    _assert(strip_overlap_prefix("其中。", "中国特色") == "中国特色", strip_overlap_prefix("其中。", "中国特色"))
    _assert(strip_overlap_prefix("自我革命", "命，提高") == "命，提高", strip_overlap_prefix("自我革命", "命，提高"))
    _assert(
        strip_overlap_prefix("自我革命", "命，提高", acoustic_overlap=True) == "提高",
        strip_overlap_prefix("自我革命", "命，提高", acoustic_overlap=True),
    )
    _assert(strip_overlap_prefix("", "今天") == "今天", "empty prev")
    _assert(strip_overlap_prefix("前文", "后文") == "后文", strip_overlap_prefix("前文", "后文"))
    _assert(strip_overlap_prefix("自我革命。", "的自我革命，提高") == "提高", strip_overlap_prefix("自我革命。", "的自我革命，提高"))
    _assert(strip_overlap_prefix("自我革命。", "命，提高") == "命，提高", strip_overlap_prefix("自我革命。", "命，提高"))
    _assert(strip_overlap_prefix("决议公布。", "决议指出，") == "决议指出，", strip_overlap_prefix("决议公布。", "决议指出，"))
    _assert(
        strip_overlap_prefix("坚定道路自信", "理论自信、制度自信") == "理论自信、制度自信",
        strip_overlap_prefix("坚定道路自信", "理论自信、制度自信"),
    )
    _assert(
        strip_overlap_prefix("创造性运用。", "造型运用是被实践证明的") == "造型运用是被实践证明的",
        strip_overlap_prefix("创造性运用。", "造型运用是被实践证明的"),
    )
    _assert(
        strip_overlap_prefix("着下班，和朋友聊天的时候刷着手机", "刷手机的时候，又焦虑着未来。")
        == "刷手机的时候，又焦虑着未来。",
        strip_overlap_prefix("着下班，和朋友聊天的时候刷着手机", "刷手机的时候，又焦虑着未来。"),
    )
    _assert(
        strip_overlap_prefix("付出巨大牺牲积累的一系列", "打牺牲积累的一系列独创性经验")
        == "独创性经验",
        strip_overlap_prefix("付出巨大牺牲积累的一系列", "打牺牲积累的一系列独创性经验"),
    )
    _assert(
        strip_overlap_prefix("策，深刻揭示社会主义本质，确立。", "制，确立社会主义初级阶段基本路线")
        == "社会主义初级阶段基本路线",
        strip_overlap_prefix("策，深刻揭示社会主义本质，确立。", "制，确立社会主义初级阶段基本路线"),
    )
    _assert(
        strip_overlap_prefix("法忍受我自己，那么这里一定有。”", "这里一定有两个存在。")
        == "两个存在。",
        strip_overlap_prefix("法忍受我自己，那么这里一定有。”", "这里一定有两个存在。"),
    )
    _assert(
        strip_overlap_prefix("我无法忍受我自己，那么这里一定有", "这里一定有两个存在。")
        == "两个存在。",
        strip_overlap_prefix("我无法忍受我自己，那么这里一定有", "这里一定有两个存在。"),
    )


def test_boundary_trim_and_stitch() -> None:
    _assert(trim_cut_fragment("决议公布。决。") == "决议公布。", trim_cut_fragment("决议公布。决。"))
    _assert(trim_cut_fragment("第一次历史性飞跃。当") == "第一次历史性飞跃。", trim_cut_fragment("第一次历史性飞跃。当"))
    _assert(trim_cut_fragment("需要。全。") == "需要。", trim_cut_fragment("需要。全。"))
    _assert(trim_cut_fragment("自我革命。") == "自我革命。", trim_cut_fragment("自我革命。"))
    _assert(trim_dangling_tail("武装夺取政权的政。") == "武装夺取政权的", trim_dangling_tail("武装夺取政权的政。"))
    _assert(stitch_prev_text("核心地位。") == "核心地位。", stitch_prev_text("核心地位。"))
    _assert(stitch_prev_text("武装夺取政权的政。") == "武装夺取政权的政。", stitch_prev_text("武装夺取政权的政。"))
    _assert(strip_boundary("自我革命。", "的自我革命，提高") == "提高", strip_boundary("自我革命。", "的自我革命，提高"))
    _assert(
        strip_boundary("武装夺取政权的政。", "政权的正确道路，创立了") == "政权的正确道路，创立了",
        strip_boundary("武装夺取政权的政。", "政权的正确道路，创立了"),
    )
    _assert(strip_boundary("文章？对。", "对。今天继续") == "今天继续", strip_boundary("文章？对。", "对。今天继续"))
    _assert(strip_boundary("我们要实现。", "现在进入新时代") == "现在进入新时代", strip_boundary("我们要实现。", "现在进入新时代"))
    _assert(strip_boundary("其中。", "中国特色社会主义") == "中国特色社会主义", strip_boundary("其中。", "中国特色社会主义"))
    _assert(strip_boundary("实现。", "现国家") == "现国家", strip_boundary("实现。", "现国家"))
    _assert(strip_boundary("决议公布。决。", "决议指出，") == "决议指出，", strip_boundary("决议公布。决。", "决议指出，"))
    _assert(
        strip_boundary("看齐意识，坚定道路自信", "理论自信、制度自信、文化自信") == "理论自信、制度自信、文化自信",
        strip_boundary("看齐意识，坚定道路自信", "理论自信、制度自信、文化自信"),
    )
    _assert(
        strip_boundary("武装夺取政权的正", "的正确革命道路，创立了") == "确革命道路，创立了",
        strip_boundary("武装夺取政权的正", "的正确革命道路，创立了"),
    )
    _assert(
        strip_boundary("为主要代表。", "主要代表的中国共产党。") == "的中国共产党。",
        strip_boundary("为主要代表。", "主要代表的中国共产党。"),
    )
    _assert(
        strip_boundary("不平等。", "不平等条约和帝国主义") == "条约和帝国主义",
        strip_boundary("不平等。", "不平等条约和帝国主义"),
    )
    _assert(
        strip_boundary("兴作为自己的初心使命，始终坚持。", "始终坚持共产主义理想") == "共产主义理想",
        strip_boundary("兴作为自己的初心使命，始终坚持。", "始终坚持共产主义理想"),
    )
    _assert(
        strip_boundary("始终坚持。", "持共产主义理想") == "持共产主义理想",
        strip_boundary("始终坚持。", "持共产主义理想"),
    )
    _assert(
        strip_boundary("平同志党中央的核心、全党的核心。", "全党的核心地位，坚决维护") == "地位，坚决维护",
        strip_boundary("平同志党中央的核心、全党的核心。", "全党的核心地位，坚决维护"),
    )
    _assert(
        strip_boundary("全党的核心。", "的核心地位，坚决维护") == "地位，坚决维护",
        strip_boundary("全党的核心。", "的核心地位，坚决维护"),
    )
    _assert(
        strip_boundary("葆党的生机活力，团结带领全国各，", "带领全国各族人民") == "族人民",
        strip_boundary("葆党的生机活力，团结带领全国各，", "带领全国各族人民"),
    )
    _assert(
        strip_boundary("策，深刻揭示社会主义本质，确立。", "制，确立社会主义初级阶段基本路线")
        == "社会主义初级阶段基本路线",
        strip_boundary("策，深刻揭示社会主义本质，确立。", "制，确立社会主义初级阶段基本路线"),
    )
    _assert(
        strip_boundary("战略、政治保证、祖国统一、外交。", "以外交和国际战略") == "和国际战略",
        strip_boundary("战略、政治保证、祖国统一、外交。", "以外交和国际战略"),
    )
    _assert(
        strip_boundary("战略、政治保证、祖国统一、外交。", "一，外交和国际战略") == "和国际战略",
        strip_boundary("战略、政治保证、祖国统一、外交。", "一，外交和国际战略"),
    )
    _assert(strip_boundary("才能发展中国。", "中国。") == "", strip_boundary("才能发展中国。", "中国。"))
    _assert(strip_boundary("赶上了时代。", "代。") == "", strip_boundary("赶上了时代。", "代。"))
    _assert(
        strip_boundary("着下班，和朋友聊天的时候刷着手机", "刷手机的时候，又焦虑着未来。")
        == "刷手机的时候，又焦虑着未来。",
        strip_boundary("着下班，和朋友聊天的时候刷着手机", "刷手机的时候，又焦虑着未来。"),
    )
    _assert(
        strip_boundary("法忍受我自己，那么这里一定有。”", "这里一定有两个存在。")
        == "两个存在。",
        strip_boundary("法忍受我自己，那么这里一定有。”", "这里一定有两个存在。"),
    )
    _assert(leftover_echo("才能发展中国。", "中国。"), "中国 echo")
    _assert(leftover_echo("赶上了时代。", "代。"), "代 echo")
    _assert(strip_forced_cut_stop("始终坚持。") == "始终坚持", strip_forced_cut_stop("始终坚持。"))
    _assert(
        strip_forced_cut_stop("时代坚持和发展中国特色社会主义。") == "时代坚持和发展中国特色社会主义",
        strip_forced_cut_stop("时代坚持和发展中国特色社会主义。"),
    )
    _assert(
        strip_forced_cut_stop("为实现中华民族伟大复兴。") == "为实现中华民族伟大复兴",
        strip_forced_cut_stop("为实现中华民族伟大复兴。"),
    )
    _assert(
        strip_forced_cut_stop("不懈奋斗。已经走过一百年光辉历程。") == "不懈奋斗。已经走过一百年光辉历程",
        strip_forced_cut_stop("不懈奋斗。已经走过一百年光辉历程。"),
    )
    _assert(
        strip_forced_cut_stop("那么这里一定有。”") == "那么这里一定有",
        strip_forced_cut_stop("那么这里一定有。”"),
    )
    _assert(not leftover_echo("伟大飞跃。", "中"), "new 中 must stay")
    _assert(dropped_is_overlap("道路自", "坚定道路自信"), "roads overlap")
    _assert(dropped_is_overlap("实现了中", "消灭一切剥削制，实现了中华"), "prefix overlap")
    _assert(not dropped_is_overlap("中", "伟大飞跃。"), "new sentence 中 must be kept")


def test_first_char_boundary_words() -> None:
    """确保 / 永葆 / 全党 / 统治: stitch must not eat a new sentence head.

    1-char strip is allowed only with an acoustic overlap marker on a mid-word cut.
    True sentence ends always need 2+ chars. Uncertain matches are kept.
    """
    _assert(
        strip_boundary("集中统一领导。", "保全党步调一致", acoustic_overlap=True)
        == "保全党步调一致",
        strip_boundary("集中统一领导。", "保全党步调一致", acoustic_overlap=True),
    )
    _assert(
        strip_boundary("应对风险挑战能力", "保党的生机活力", acoustic_overlap=True)
        == "保党的生机活力",
        strip_boundary("应对风险挑战能力", "保党的生机活力", acoustic_overlap=True),
    )
    _assert(
        strip_boundary("继续奋斗，需要。", "党要坚持唯物史观", acoustic_overlap=True)
        == "党要坚持唯物史观",
        strip_boundary("继续奋斗，需要。", "党要坚持唯物史观", acoustic_overlap=True),
    )
    _assert(
        strip_boundary("剥削者统", "统治广大劳动人民", acoustic_overlap=True)
        == "治广大劳动人民",
        strip_boundary("剥削者统", "统治广大劳动人民", acoustic_overlap=True),
    )
    _assert(
        strip_boundary("剥削者统", "统治广大劳动人民") == "统治广大劳动人民",
        strip_boundary("剥削者统", "统治广大劳动人民"),
    )
    _assert(
        strip_boundary("剥削者统", "广大劳动人民", acoustic_overlap=True)
        == "广大劳动人民",
        strip_boundary("剥削者统", "广大劳动人民", acoustic_overlap=True),
    )


def test_soft_cut_split_not_full_copy() -> None:
    sr = 16000
    start = 187.44
    resume = 197.66
    buf_end = 197.68
    n = int(round((buf_end - start) * sr))
    buf = np.arange(n, dtype=np.float32)
    plan = plan_soft_cut(start, resume, 380.0, 200.0, buf_end)
    _assert(plan.next_start < plan.silence_start, plan)
    _assert(plan.silence_start < plan.previous_end < plan.resume_time, plan)
    _assert(plan.overlap_seconds > 0.1, plan)
    prev, state = split_soft_cut(buf, start, plan, sr)
    full_s = buf.size / sr
    prev_s = prev.size / sr
    next_s = state.speech_buf.size / sr
    _assert(prev_s < full_s - 0.05, (prev_s, full_s))
    _assert(next_s > 0.30, next_s)
    _assert(abs(state.drop_before - plan.previous_end) < 1e-6, state)
    _assert(abs(state.buffer_audio_start - plan.next_start) < 1e-6, state)
    _assert(abs(state.commit_start - plan.previous_end) < 1e-6, state)
    lookback = min(0.2, plan.silence_start - start)
    _assert(abs((plan.silence_start - plan.next_start) - lookback) < 1e-6, plan)


def test_vad_turn_overlap_independent_of_hard_cut() -> None:
    sr = 16000
    buf = np.ones(int(2.0 * sr), dtype=np.float32)
    state = apply_hard_cut(buf, 10.0, 0.4, sr, 0.0)
    _assert(abs(state.speech_buf.size / sr - 0.4) < 1e-6, state.speech_buf.size / sr)
    _assert(abs(state.buffer_audio_start - 11.6) < 1e-6, state.buffer_audio_start)
    zero = apply_hard_cut(buf, 10.0, 0.0, sr, 0.0)
    _assert(zero.speech_buf.size == 0, zero.speech_buf.size)


def test_text_ends_sentence() -> None:
    _assert(text_ends_sentence("决议公布。"), "period")
    _assert(text_ends_sentence("你好吗？"), "question")
    _assert(text_ends_sentence("快走！"), "bang")
    _assert(text_ends_sentence("说完了。”"), "quote after period")
    _assert(text_ends_sentence("好的。"), "short meeting ack")
    _assert(text_ends_sentence("知道了。"), "short meeting ack 3")
    _assert(text_ends_sentence("没问题。"), "short meeting ack 3")
    _assert(text_ends_sentence("让我国外贸抵御风险能力更强。"), "long complete")
    _assert(not text_ends_sentence("增速高出0.4个百"), "mid number")
    _assert(not text_ends_sentence("海南免税消费稳步"), "mid phrase")
    _assert(not text_ends_sentence("五、黄河山西段开鱼"), "list item")
    _assert(not text_ends_sentence(""), "empty")
    _assert(not text_ends_sentence("好的"), "short no punct")
    _assert(not text_ends_sentence("有助于带动沿海城市增收、吸纳就业，蓝色海洋。"), "fake period short clause")
    _assert(not text_ends_sentence("近海养殖、港口航运、海上风电、滨海等。"), "fake period 等")
    _assert(text_ends_sentence("增长五点一个百分点，比国内整体。"), "last clause >4 CJK is structural end")
    _assert(text_ends_sentence("武装夺取政权的政。"), "no char-class; last clause is long")
    _assert(not text_ends_sentence("做到坚决维护党中央权威和集中统一领导，确。"), "comma + 1 CJK period")
    _assert(not text_ends_sentence("反对帝国主义、封建主义、官僚。"), "dunhao + 2 CJK period")


def test_fresh_audio_samples() -> None:
    sr = 16000
    _assert(fresh_audio_samples(20 * sr, 0, 0, sr) == 20 * sr, "ordinary hard cut")
    _assert(fresh_audio_samples(20 * sr, 19.2, 20, sr) == int(19.2 * sr), "context must not age a turn")
    _assert(fresh_audio_samples(int(20.8 * sr), 19.2, 20, sr) == 20 * sr, "20s fresh audio")


def test_hard_cut_holdback_moves_commit() -> None:
    restored = finalize_ws_utterance(
        [
            WordTimestamp("中", 0.16, 0.40),
            WordTimestamp("国", 0.70, 0.90),
        ],
        "中国",
        audio_start=310.0,
        audio_duration=4.0,
        commit_start=310.60,
        drop_before=310.60,
        eps=0.15,
        overlap_prefix="伟大飞跃。",
    )
    _assert([w.word for w in restored[3]] == ["中", "国"], restored)
    stale_prev = "创造根本社会条件。在革命斗争中，"
    long_overlap = finalize_ws_utterance(
        [
            WordTimestamp("以", 0.05, 0.10),
            WordTimestamp("毛", 0.10, 0.15),
            WordTimestamp("泽", 0.15, 0.20),
            WordTimestamp("东", 0.20, 0.25),
            WordTimestamp("同", 0.25, 0.30),
            WordTimestamp("志", 0.30, 0.35),
            WordTimestamp("为", 0.35, 0.40),
            WordTimestamp("主", 0.55, 0.60),
            WordTimestamp("要", 0.60, 0.65),
            WordTimestamp("代", 0.65, 0.70),
            WordTimestamp("表", 0.70, 0.75),
            WordTimestamp("的", 0.75, 0.80),
            WordTimestamp("中", 0.80, 0.85),
            WordTimestamp("国", 0.85, 0.90),
            WordTimestamp("共", 0.90, 0.95),
            WordTimestamp("产", 0.95, 1.00),
            WordTimestamp("党", 1.00, 1.05),
        ],
        "以毛泽东同志为主要代表的中国共产党",
        audio_start=20.0,
        audio_duration=1.2,
        commit_start=20.50,
        drop_before=20.50,
        eps=0.15,
        overlap_prefix=stale_prev,
    )
    _assert(long_overlap[0].startswith("以毛泽东"), long_overlap[0])
    _assert("中国共产党" in long_overlap[0], long_overlap[0])
    dropped = [
        WordTimestamp("以", 20.05, 20.10),
        WordTimestamp("毛", 20.10, 20.15),
        WordTimestamp("泽", 20.15, 20.20),
        WordTimestamp("东", 20.20, 20.25),
        WordTimestamp("同", 20.25, 20.30),
        WordTimestamp("志", 20.30, 20.35),
        WordTimestamp("为", 20.35, 20.40),
    ]
    kept = [
        WordTimestamp("主", 20.55, 20.60),
        WordTimestamp("要", 20.60, 20.65),
        WordTimestamp("代", 20.65, 20.70),
        WordTimestamp("表", 20.70, 20.75),
    ]
    restored_long = restore_nonoverlap_dropped(dropped + kept, kept, stale_prev)
    _assert([w.word for w in restored_long] == ["主", "要", "代", "表"], restored_long)
    three = [
        WordTimestamp("始", 21.48, 21.64),
        WordTimestamp("终", 21.64, 21.80),
        WordTimestamp("坚", 21.80, 22.04),
    ]
    kept_hold = [
        WordTimestamp("持", 22.20, 22.44),
        WordTimestamp("共", 22.44, 22.60),
    ]
    no_restore = restore_nonoverlap_dropped(three + kept_hold, kept_hold, "懈奋斗。已经走过一百年光辉历程。")
    _assert([w.word for w in no_restore] == ["持", "共"], no_restore)
    persist = finalize_ws_utterance(
        [
            WordTimestamp("始", 0.24, 0.40),
            WordTimestamp("终", 0.40, 0.56),
            WordTimestamp("坚", 0.56, 0.80),
            WordTimestamp("持", 1.00, 1.20),
            WordTimestamp("共", 1.20, 1.36),
            WordTimestamp("产", 1.36, 1.52),
        ],
        "始终坚持共产",
        audio_start=21.24,
        audio_duration=2.0,
        commit_start=22.24,
        drop_before=22.24,
        eps=0.15,
        overlap_prefix="兴作为自己的初心使命，始终坚持。",
    )
    _assert(not persist[0].startswith("始"), persist[0])
    _assert("共产" in persist[0], persist[0])
    core_repeat = finalize_ws_utterance(
        [
            WordTimestamp("全", 0.00, 0.48),
            WordTimestamp("党", 0.48, 0.56),
            WordTimestamp("的", 0.56, 0.72),
            WordTimestamp("核", 0.72, 0.88),
            WordTimestamp("心", 0.88, 1.04),
            WordTimestamp("地", 1.04, 1.20),
            WordTimestamp("位", 1.20, 1.36),
        ],
        "全党的核心地位",
        audio_start=75.72,
        audio_duration=2.0,
        commit_start=76.72,
        drop_before=76.72,
        eps=0.15,
        overlap_prefix="平同志党中央的核心、全党的核心。",
    )
    _assert(core_repeat[0] == "地位", core_repeat[0])
    echo = finalize_ws_utterance(
        [
            WordTimestamp("中", 0.00, 0.16),
            WordTimestamp("国", 0.16, 0.16),
            WordTimestamp("。", 0.16, 0.16),
        ],
        "中国。",
        audio_start=328.82,
        audio_duration=2.92,
        commit_start=329.82,
        drop_before=329.82,
        eps=0.15,
        overlap_prefix="只有社会主义才能发展中国。",
    )
    _assert(echo[0] == "", echo[0])
    era = finalize_ws_utterance(
        [
            WordTimestamp("赶", 0.08, 0.32),
            WordTimestamp("上", 0.40, 0.56),
            WordTimestamp("了", 0.56, 0.64),
            WordTimestamp("时", 0.64, 0.96),
            WordTimestamp("代", 1.00, 1.04),
            WordTimestamp("。", 1.04, 1.08),
        ],
        "赶上了时代。",
        audio_start=618.32,
        audio_duration=2.78,
        commit_start=619.32,
        drop_before=619.32,
        eps=0.15,
        overlap_prefix="中国大踏步赶上了时代。",
    )
    _assert(era[0] == "", era[0])


def test_final_overlap_text_first() -> None:
    """Atomic whole-span delete; timestamps confirm; 1-char and 确保 stay."""
    bureau = finalize_ws_utterance(
        [
            WordTimestamp("官", 0.16, 0.32),
            WordTimestamp("僚", 0.32, 0.50),
            WordTimestamp("资", 0.50, 0.66),
            WordTimestamp("本", 0.66, 0.82),
        ],
        "官僚资本主义",
        audio_start=161.62,
        audio_duration=2.0,
        commit_start=162.02,
        drop_before=162.02,
        eps=0.15,
        overlap_prefix="反对帝国主义、封建主义、官僚。",
    )
    _assert(bureau[0] == "资本主义", bureau[0])
    _assert([w.word for w in bureau[3]] == ["资", "本"], bureau[3])
    _assert(bureau[4].removed_plain == "官僚", bureau[4])
    half = finalize_ws_utterance(
        [
            WordTimestamp("是", 0.16, 0.32),
            WordTimestamp("在", 0.32, 0.48),
            WordTimestamp("建", 0.50, 0.66),
            WordTimestamp("党", 0.66, 0.82),
        ],
        "是在建党",
        audio_start=77.16,
        audio_duration=2.0,
        commit_start=77.56,
        drop_before=77.56,
        eps=0.15,
        overlap_prefix="历史经验，是在。",
    )
    _assert(half[0] == "建党", half[0])
    _assert(half[4].removed_plain == "是在", half[4])
    ensure = finalize_ws_utterance(
        [
            WordTimestamp("确", 0.42, 0.58),
            WordTimestamp("保", 0.58, 0.74),
            WordTimestamp("全", 0.74, 0.90),
            WordTimestamp("党", 0.90, 1.06),
        ],
        "确保全党",
        audio_start=108.24,
        audio_duration=2.0,
        commit_start=108.64,
        drop_before=108.64,
        eps=0.15,
        overlap_prefix="集中统一领导，确。",
    )
    _assert(ensure[0] == "确保全党", ensure[0])
    _assert(ensure[4].match_n == 1, ensure[4])
    _assert(ensure[4].removed_plain == "", ensure[4])
    central = finalize_ws_utterance(
        [
            WordTimestamp("中", 0.00, 0.16),
            WordTimestamp("央", 0.16, 0.16),
            WordTimestamp("关", 0.40, 0.56),
            WordTimestamp("于", 0.56, 0.72),
        ],
        "中央关于",
        audio_start=31.94,
        audio_duration=2.0,
        commit_start=32.34,
        drop_before=32.34,
        eps=0.15,
        overlap_prefix="清楚。",
    )
    _assert(central[0] == "中央关于", central[0])
    _assert(central[4].removed_plain == "", central[4])
    now = finalize_ws_utterance(
        None,
        "现在进入新时代",
        audio_start=10.0,
        audio_duration=2.0,
        commit_start=10.4,
        drop_before=10.4,
        eps=0.15,
        overlap_prefix="我们要实现。",
    )
    _assert(now[0] == "现在进入新时代", now[0])
    no_words = finalize_ws_utterance(
        None,
        "官僚资本主义",
        audio_start=161.62,
        audio_duration=2.0,
        commit_start=162.02,
        drop_before=162.02,
        eps=0.15,
        overlap_prefix="封建主义、官僚。",
    )
    _assert(no_words[0] == "资本主义", no_words[0])
    no_acoust = finalize_ws_utterance(
        None,
        "官僚资本主义",
        audio_start=162.02,
        audio_duration=2.0,
        commit_start=162.02,
        drop_before=None,
        eps=0.15,
        overlap_prefix="封建主义、官僚。",
    )
    _assert(no_acoust[0] == "官僚资本主义", no_acoust[0])
    punct_raw = finalize_ws_utterance(
        None,
        "，官僚资本主义",
        audio_start=161.62,
        audio_duration=2.0,
        commit_start=162.02,
        drop_before=162.02,
        eps=0.15,
        overlap_prefix="封建主义、官僚。",
    )
    _assert(punct_raw[0] == "资本主义", punct_raw[0])


def test_vad_overlap_reties_to_stream() -> None:
    sr = 16000
    origin = realtime_buffer_origin(int(24.0 * sr), int(2.0 * sr), sr)
    _assert(abs(origin - 22.0) < 1e-6, origin)
    # Cut at 20s, keep 1s + 0.8s hangover, skip 3s silence, then a 0.2s packet.
    start, boundary = retie_turn_overlap(int(1.8 * sr), int(0.2 * sr), int(24.0 * sr), sr)
    _assert(abs(start - 22.0) < 1e-6, start)
    _assert(abs(boundary - 23.8) < 1e-6, boundary)
    words = [
        WordTimestamp("旧", 0.40, 0.55),
        WordTimestamp("句", 0.55, 0.70),
        WordTimestamp("新", 1.90, 2.00),
    ]
    text, _s, _e, kept, _dec = finalize_ws_utterance(
        words,
        "旧句新",
        audio_start=start,
        audio_duration=2.0,
        commit_start=boundary,
        drop_before=boundary,
        eps=0.15,
        overlap_prefix="上一句旧句。",
    )
    _assert(text == "新", text)
    _assert(abs(kept[0].start - 23.9) < 1e-6, kept[0])
    # No gap: hangover promote must still land on the original cut.
    cont_start, cont_cut = retie_turn_overlap(sr, int(0.6 * sr), int(20.6 * sr), sr)
    _assert(abs(cont_start - 19.0) < 1e-6, cont_start)
    _assert(abs(cont_cut - 20.0) < 1e-6, cont_cut)


def test_pre_roll_unused_excludes_current_packet() -> None:
    max_n = 6400
    ring = np.zeros(0, dtype=np.float32)
    p1 = np.ones(1000, dtype=np.float32)
    ring = append_pre_roll(ring, p1, max_n)
    p2 = np.full(1000, 2.0, dtype=np.float32)
    ring = append_pre_roll(ring, p2, max_n)
    unused = unused_pre_roll(ring, p2)
    _assert(unused.size == 1000, str(unused.size))
    _assert(float(unused[0]) == 1.0, str(unused[0]))
    big = np.ones(8000, dtype=np.float32)
    ring = append_pre_roll(np.zeros(0, dtype=np.float32), big, max_n)
    _assert(unused_pre_roll(ring, big).size == 0, "current packet already covers the ring")


def test_realtime_one_window() -> None:
    _assert(realtime_window_sec(3.0, 8.0) == 8.0, realtime_window_sec(3.0, 8.0))
    _assert(abs(realtime_window_sec(18.0, 8.0) - 18.05) < 1e-9, realtime_window_sec(18.0, 8.0))
    src = (Path(__file__).resolve().parents[1] / "src" / "backend" / "gguf_backend.py").read_text(encoding="utf-8")
    rt = src.split("def transcribe_realtime")[1].split("def unload")[0]
    _assert("realtime_window_sec" in rt, rt)
    _assert("WS_EMPTY_RESULT" in rt, "empty realtime result must log WS_EMPTY_RESULT instead of retrying")
    _assert("_once(REALTIME_CHUNK_SIZE_SEC" not in rt, "same-param retry removed: no second _once(REALTIME_CHUNK_SIZE_SEC) call")
    _assert('_once(window, "")' in rt, "first realtime pass must not use decoder prefix")
    _assert("WS_FINAL_ALIGN" in rt, "realtime final align must honor WS_FINAL_ALIGN")


def test_pending_caption_prefix() -> None:
    shown = {
        1: "中坚持共产主义理想和社会主义信念，团结带领全国各族人民。",
        2: "党和人民百年奋斗。",
    }
    held = pending_caption_prefix(shown, set(), 2)
    _assert(held == shown[1], held)
    _assert(pending_caption_prefix(shown, {1}, 2) == "", "finalized previous is not held")
    _assert(
        pending_caption_prefix(shown, set(), 3) == shown[1] + shown[2],
        "two unfinalized predecessors stay on screen",
    )
    src = (SRC / "server.py").read_text(encoding="utf-8")
    _assert("pending_caption_prefix" in src and "WS partial_hold" in src, src)


def test_join_partial_tail() -> None:
    prev = "本台消息，十一月十六号，中共。"
    tail = "消息：十一月十六号，中共中央关于党的百。"
    joined = join_partial_tail(prev, tail)
    _assert(joined == prev + tail, "fresh PCM chunks must append without text dedupe")
    _assert(join_partial_tail("", "新句子") == "新句子", "empty prev")
    _assert(join_partial_tail(prev, "") == prev, "empty tail")
    shown = "已经显示的正文内容"
    _assert(
        join_partial_tail(shown, "内容后续还没对上") == shown + "内容后续还没对上",
        "no overlap appends the fresh window",
    )
    stable = "前面稳定的内容关于党的百年奋斗"
    _assert(
        join_partial_tail(stable, "关于党的百年奋斗重大成就")
        == stable + "关于党的百年奋斗重大成就",
        "matching characters must not trigger text rewriting",
    )
    _assert(
        join_partial_tail("族几千年历史上最恢宏的史诗。", "总结党的百年奋斗。")
        == "族几千年历史上最恢宏的史诗。总结党的百年奋斗。",
        "fresh windows append directly",
    )


def test_partial_terminal_punctuation() -> None:
    pieces = ["是推进党。", "的自我革命。", "提高全党斗。", "斗争本领和应。", "对风险挑战。"]
    shown = ""
    for piece in pieces:
        shown = join_partial_tail(shown, strip_partial_terminal_punct(piece))
    _assert(shown == "是推进党的自我革命提高全党斗斗争本领和应对风险挑战", shown)
    _assert(strip_partial_terminal_punct("革命时期，党面临的主") == "革命时期，党面临的主", "internal comma")
    _assert(strip_partial_terminal_punct("反对帝国主义、封建。") == "反对帝国主义、封建", "dunhao")
    _assert(strip_partial_terminal_punct("他说：“你好。”") == "他说：“你好”", "trailing quote")
    _assert(strip_partial_terminal_punct("好！?") == "好", "all terminal stops")
    _assert(strip_partial_terminal_punct("。") == "", "punct-only partial")


def test_partial_request_gate() -> None:
    gate = PartialRequestGate()
    _assert(not gate.defer_if_running(1), "first partial may start")
    gate.started(1)
    _assert(gate.defer_if_running(1), "newer audio waits for current result")
    _assert(gate.defer_if_running(1), "many updates collapse into one pending run")
    _assert(gate.finished(1, 1), "same utterance must run the pending fresh audio")
    _assert(not gate.defer_if_running(1), "completed partial releases the gate")
    gate.started(1, more_audio=True)
    _assert(gate.finished(1, 1), "audio beyond the window cap must be drained")
    gate.started(1)
    gate.defer_if_running(1)
    _assert(not gate.finished(1, 2), "a cut must not restart the old utterance")
    gate.started(2)
    gate.defer_if_running(2)
    gate.reset()
    _assert(not gate.finished(2, 2), "reset discards a pending partial")


def test_ordered_final_buffer() -> None:
    finals = OrderedFinalBuffer()
    _assert(finals.put(3, {"start": 64.0, "text": "增强政治意识"}) == [], "utt 3 must wait")
    _assert(finals.put(2, {"start": 44.0, "text": "族几千年"}) == [], "utt 2 must wait for 0/1")
    _assert([uid for uid, _ in finals.put(0, {"start": 0.0})] == [0], "first final")
    ready = finals.put(1, {"start": 24.0})
    _assert([uid for uid, _ in ready] == [1, 2, 3], ready)
    _assert([item["start"] for _, item in ready] == [24.0, 44.0, 64.0], ready)
    _assert(finals.put(2, {"start": 44.0}) == [], "late duplicate must not re-send")
    _assert(finals.put(5, {"start": 87.0}) == [], "future final waits")
    _assert([uid for uid, _ in finals.put(4, None)] == [4, 5], "suppressed final cannot block later turns")
    finals.reset(8)
    _assert(finals.next_utt_id == 8 and not finals.pending, "reset skips discarded turn")
    _assert([uid for uid, _ in finals.put(8, {"start": 100.0})] == [8], "after reset")


def test_nonoverlapping_partial_backlog() -> None:
    sr = 16000
    cap = 4 * sr
    origin = 23.58
    buffered = 9 * sr
    gate = PartialRequestGate()
    cursor = None
    block_start = origin
    ranges: list[tuple[int, int]] = []
    while True:
        plan = plan_partial_window(
            buffered, cap, has_shown=cursor is not None,
            sample_rate=sr, origin_start=origin, prev_window_end=cursor,
            block_start=block_start,
        )
        _assert(plan is not None and plan.keep > 0, "backlog must have a fresh range")
        ranges.append((plan.offset, plan.offset + plan.keep))
        gate.started(1, more_audio=plan.step)
        _assert(gate.defer_if_running(1), "new packets must not submit a duplicate range")
        cursor = origin + ranges[-1][1] / sr
        if plan.complete:
            block_start = cursor
        should_continue = gate.finished(1, 1)
        if ranges[-1][1] == buffered:
            break
        _assert(should_continue, "backlog must be scheduled for another partial")
    _assert(ranges == [(0, 4 * sr), (4 * sr, 8 * sr), (8 * sr, 9 * sr)], ranges)
    _assert(join_partial_tail("百年奋斗", "重大成就") == "百年奋斗重大成就", "completed blocks append")


def test_partial_window_keeps_head_before_first_emit() -> None:
    sr = 16000
    n = int(7.0 * sr)
    cap = int(4.0 * sr)
    head = plan_partial_window(n, cap, has_shown=False)
    _assert(head is not None and head.offset == 0 and head.keep == cap and head.head and head.step and head.complete, head)
    tail = plan_partial_window(n, cap, has_shown=True)
    _assert(tail is None, "cannot advance a shown transcript without its audio cursor")
    at_cap = plan_partial_window(cap, cap, has_shown=False)
    _assert(at_cap is not None and at_cap.complete and at_cap.keep == cap, at_cap)
    first = plan_partial_window(int(1.2 * sr), cap, has_shown=False)
    _assert(first is not None and first.offset == 0 and first.keep == int(1.2 * sr), first)
    origin = 124.22
    revised = plan_partial_window(
        n, cap, has_shown=True, sample_rate=sr, origin_start=origin,
        prev_window_end=origin + 1.2, block_start=origin,
    )
    _assert(revised is not None and revised.offset == 0 and revised.keep == cap and revised.complete, revised)
    _assert(plan_partial_window(
        int(1.2 * sr), cap, has_shown=True, sample_rate=sr,
        origin_start=origin, prev_window_end=origin + 1.2, block_start=origin,
    ).keep == 0, "same audio must not submit twice")
    after_head = plan_partial_window(
        n, cap, has_shown=True, sample_rate=sr, origin_start=origin,
        prev_window_end=origin + 4.0, block_start=origin + 4.0,
    )
    _assert(after_head is not None and after_head.offset == int(4.0 * sr), after_head)
    _assert(after_head.keep == n - after_head.offset and not after_head.head, after_head)
    caught_up = plan_partial_window(
        int(8.0 * sr), cap, has_shown=True, sample_rate=sr,
        origin_start=origin, prev_window_end=origin + 6.0, block_start=origin + 4.0,
    )
    _assert(caught_up is not None and caught_up.offset == int(4.0 * sr), caught_up)
    _assert(caught_up.keep == cap and caught_up.complete and not caught_up.step, caught_up)
    behind = plan_partial_window(
        int(10.0 * sr), cap, has_shown=True, sample_rate=sr,
        origin_start=origin, prev_window_end=origin + 4.0, block_start=origin + 4.0,
    )
    _assert(behind is not None and behind.offset == int(4.0 * sr) and behind.keep == cap and behind.step, behind)
    growing = plan_partial_window(
        int(2.4 * sr), cap, has_shown=True, sample_rate=sr,
        origin_start=origin, prev_window_end=origin + 1.2, block_start=origin,
    )
    _assert(growing is not None and growing.offset == 0, growing)
    _assert(growing.keep == int(2.4 * sr), growing)
    src = (SRC / "server.py").read_text(encoding="utf-8")
    _assert("prev_window_end=" in src, "emit_asr must pass previous window end")
    _assert("block_start=" in src, "emit_asr must retain the open block start")
    _assert("partial_committed_text.get(job_utt_id" in src, "open block text must replace its prior version")
    _assert("partial_block_start[job_utt_id] = last_partial_audio_end[job_utt_id]" in src,
            "only a completed block may advance the no-overlap boundary")
    _assert("partial_window" in src, src)
    _assert("window.keep <= 0" in src, "fresh-window planner must skip empty slices")
    _assert("partial_gate.defer_if_running" in src, "one partial at a time per utterance")
    _assert("partial_gate.finished" in src, "deferred audio must resume after inference")
    _assert("has_shown=job_utt_id in last_partial_audio_end" in src, "empty partial still advances PCM")


def test_fallback_empty_final() -> None:
    _assert(fallback_empty_final("", "", "") == "", "no fallback")
    _assert(fallback_empty_final("", "", "上一句部分") == "上一句部分", "use last partial")
    _assert(
        fallback_empty_final("", "", "对。今天继续", overlap_prefix="文章？对。", drop_before=20.0) == "今天继续",
        fallback_empty_final("", "", "对。今天继续", overlap_prefix="文章？对。", drop_before=20.0),
    )
    _assert(fallback_empty_final("有字", "", "上一句") == "", "finalize emptied on purpose")


def test_final_reads_cache_before_drop() -> None:
    src = (SRC / "server.py").read_text(encoding="utf-8")
    begin_fn = src.split("def _begin_next_utterance")[1].split("def _apply_hard_cut")[0]
    hard_fn = src.split("def _apply_hard_cut(")[1].split("def _apply_soft_cut")[0]
    soft_fn = src.split("def _apply_soft_cut")[1].split("def _apply_vad_turn")[0]
    vad_fn = src.split("def _apply_vad_turn")[1].split("async def _run_and_send")[0]
    run_fn = src.split("async def _run_and_send")[1].split("def _spawn")[0]
    for name, body in (
        ("begin", begin_fn),
        ("hard", hard_fn),
        ("soft", soft_fn),
        ("vad", vad_fn),
    ):
        _assert("drop_utt_cache" not in body, f"{name} drops cache before the final runs")
    asr_at = run_fn.find("await _run_asr")
    drop_at = run_fn.find("drop_utt_cache")
    _assert(asr_at >= 0 and drop_at > asr_at, "final must drop the utt cache only after ASR")
    _assert('if is_final and result is not None' in run_fn, run_fn[drop_at - 80:drop_at + 80])


def test_hard_cut_does_not_reset_vad() -> None:
    src = (Path(__file__).resolve().parents[1] / "src" / "server.py").read_text(encoding="utf-8")
    _assert("apply_hard_cut(" in src, "hard cut helper missing")
    _assert("allow_final_yield=not (is_final and hard_cut)" in src,
            "hard-cut final must finish before next-turn partial refresh")
    _assert("if yield_event is None and allow_final_yield:" in src,
            "disabled yield must not create a backend-visible event")
    _assert("unused_pre_roll(" in src, "pre-roll unused missing")
    apply_fn = src.split("def _apply_hard_cut(")[1].split("async def _run_and_send")[0]
    _assert("vad.reset()" not in apply_fn, apply_fn)
    begin_fn = src.split("def _begin_next_utterance")[1].split("def _apply_hard_cut")[0]
    _assert("if reset_vad:" in begin_fn, begin_fn)
    _assert("def _apply_vad_turn(" in src, "vad turn helper missing")
    vad_fn = src.split("def _apply_vad_turn")[1].split("async def _run_and_send")[0]
    _assert("keep_overlap" not in vad_fn, "vad turn no longer keeps previous-turn PCM")
    _assert("WS_VAD_TURN_OVERLAP_SECONDS" not in src, "VAD turn overlap config is gone")
    _assert("turn_overlap" in src, "hangover overlap buffer missing")
    _assert("vad.reset()" not in src.split("def _apply_vad_turn")[1].split("async def _run_and_send")[0], "vad_end must not reset VAD")
    _assert("carry_from_utt == job_utt_id - 1" in src, "stitch committed prefix only when it is the previous utt")
    _assert("job_utt_id >= carry_from_utt" in src, "late previous final must still update carry")
    _assert("overlap_echo" in src, "must drop leftover echo finals")
    _assert("strip_forced_cut_stop(" in src, "hard cut must drop decoder trailing 。")
    _assert("speculative_prefix = carry_prefix_tail(stitch_prev_text(guess)" in src, "cut guess is for the next utt only")
    _assert("carry_prefix = carry_prefix_tail(stitch_prev_text(guess)" not in src, "must not overwrite committed prefix with last partial")
    _assert("kept_hangover" in src, "hangover discard must keep trailing PCM")
    _assert("plan_soft_cut(" in src, "soft-cut must plan previous_end/next_start")
    _assert("split_soft_cut(" in src, "soft-cut must split previous/next PCM")
    _assert("audio = speech_buf.copy()" in src.split("async def emit_asr")[1].split("task = _spawn")[0], src)
    _assert("audio = _apply_soft_cut(plan)" in src, "previous soft-cut submit must use split audio, not the full buffer")
    _assert("overlap_actual_remove" in src, "final overlap must log the atomic remove")
    _assert("hold_cut_tail(" not in src, "vad_end must not hold unstable tail chars")
    _assert("WS boundary_stitch" not in src, "final must not strip_boundary a second time")
    _assert("WS_TRIM_CUT_FRAGMENT" not in src, "realtime must not trim by character class")
    _assert("trim_dangling_tail(" not in src, "realtime must not call trim_dangling_tail")
    _assert("WS boundary_trim" not in src, "realtime must not rewrite finals after overlap")
    vad_end = src.split("if use_vad and (event.speech_ended or force_speech_ended):", 1)[1].split(
        "# A resumed pause can soft-cut", 1
    )[0]
    _assert("punct_defer = True" in vad_end, "all VAD endings must enter the bounded resume grace")
    _assert("await emit_asr(True, reason=\"vad_end\")" not in vad_end,
            "partial punctuation must not bypass the resume grace")
    defer = src.split("if punct_defer and not event.is_speech:", 1)[1].split("prev_energy_run", 1)[0]
    _assert("if punct_defer_silence >= punct_defer_cap:" in defer,
            "resume grace must end only at its fixed silence cap")
    _assert("punct_ready or punct_defer_silence" not in defer,
            "partial punctuation must not shorten the resume grace")
    _assert("if (not endpoint_resumed" in src,
            "speech resuming during endpoint grace must not immediately soft-cut")
    _assert("vad_defer" in src, "incomplete utterances must defer VAD cut")
    _assert("silence_cap" in src, "deferred VAD cut must expire on extra silence")
    _assert("choose_hard_cut_silence(" in src, "hard cut must search VAD silence first")
    _assert("if utterance_duration >= WS_SOFT_CUT_START_SECONDS:" in src,
            "soft cut must be allowed at a qualifying pause even without punctuation")
    _assert("WS_HARD_CUT_OVERLAP_SECONDS" in src,
            "fallback hard cut must preserve PCM overlap for the next utterance")
    _assert("retie_turn_overlap(" in src, "VAD overlap onset must retie timestamps to the stream")
    _assert("discard_stale_overlap_after_skip(" in src, "pause after VAD must not reuse previous-turn PCM")
    _assert("onset_fresh" in src, "pause onset must log discarded stale overlap")
    _assert("pieces.append(unused)" not in src, "must not stack pre-roll after turn overlap")
    enc = (Path(__file__).resolve().parents[1] / "vendor" / "qwen_asr_gguf" / "inference" / "encoder.py").read_text(encoding="utf-8")
    _assert("整数倍" in enc, enc)


def test_encoder_cache_and_commit() -> None:
    sr = 16000
    _assert(not is_full_window(3 * sr, 8 * sr), "short clip is not a full window")
    _assert(is_full_window(8 * sr, 8 * sr), "exact 8s slice is a full window")
    _assert(not is_full_window(8 * sr - 1, 8 * sr), "tail shorter than 8s is not full")
    _assert(
        not chunk_commit_ready(0, chunk_size_sec=8.0, total_duration=8.0, margin_sec=0.4, was_last=False, full_window=True),
        "8.0s has not passed the 400ms margin",
    )
    _assert(
        chunk_commit_ready(0, chunk_size_sec=8.0, total_duration=8.4, margin_sec=0.4, was_last=False, full_window=True),
        "8.4s may commit the first full window",
    )
    _assert(
        not chunk_commit_ready(1, chunk_size_sec=8.0, total_duration=16.4, margin_sec=0.4, was_last=True, full_window=True),
        "last window is never committed",
    )
    cache = {}
    set_cache_embd(cache, 0, "embd")
    _assert(should_skip_decode(cache, 0, was_last=False, enabled=True) == (False, None), "embd alone does not skip")
    commit_chunk_text(cache, 0, "百年")
    _assert(should_skip_decode(cache, 0, was_last=False, enabled=True) == (True, "百年"), should_skip_decode(cache, 0, was_last=False, enabled=True))
    _assert(should_skip_decode(cache, 0, was_last=True, enabled=True) == (False, None), "last window always decodes")
    _assert(should_skip_decode(cache, 0, was_last=False, enabled=False) == (False, None), "skip stays off until the flag")
    store = UtteranceEncoderCache(fingerprint="fp")
    first = store.window("s", 1, 0)
    first[0] = {"embd": "a"}
    again = store.window("s", 1, 0)
    _assert(again.get(0, {}).get("embd") == "a", "same start keeps the bucket")
    moved = store.window("s", 1, 16000)
    _assert(0 not in moved, "audio start change drops the old bucket")


def test_clocks_and_timing_regex() -> None:
    from benchmark_ws import CLOCKS_RE, TIMING_RE

    line = (
        "WS timing | kind=final reason=hard_cut utt=0 audio_end=23.580 "
        "audio_dur=20.000s total_ms=7548.2 queue_ms=3.7 infer_ms=7543.6 "
        "encode_ms=1791.6 prefill_ms=37.0 decode_ms=5609.4 n_chunks=3 "
        "cache_hits=0 decode_skipped=0 result=ready"
    )
    match = TIMING_RE.search(line)
    _assert(match is not None, "timing regex must parse encode/decode/cache fields")
    _assert(match.group(1) == "final", match.group(1))
    _assert(match.group(6) == "1791.6", match.group(6))
    _assert(match.group(8) == "5609.4", match.group(8))
    _assert(match.group(10) == "0", match.group(10))
    clock = (
        "WS clocks | kind=partial reason=partial utt=1 infer_done=1 ws_emit=1 "
        "text_changed=1 endpoint_wait_ms=0.0 final_refresh_gap_s=9.640"
    )
    cmatch = CLOCKS_RE.search(clock)
    _assert(cmatch is not None, clock)
    _assert(cmatch.group(2) == "1", cmatch.group(2))
    _assert(cmatch.group(3) == "9.640", cmatch.group(3))
    src = (SRC / "server.py").read_text(encoding="utf-8")
    _assert("WS clocks |" in src, "server must emit three-clock lines")
    _assert("text_changed" in src, "server must record whether emitted text changed")
    _assert("final_refresh_gap_s" in src, "server must record the post-final refresh gap")


def test_shared_utt_origin_not_wiped() -> None:
    store = UtteranceEncoderCache(fingerprint="fp")
    origin = store.ensure_origin("s", 1, 0)
    _assert(origin is not None, "ensure_origin creates the origin bucket")
    origin[0] = {"embd": "keep"}
    _assert(store.peek_chunks("s", 1, 0).get(0, {}).get("embd") == "keep", "peek same origin")
    _assert(store.peek_chunks("s", 1, 64000) is None, "peek at a sliding start must not create or wipe")
    _assert(store.ensure_origin("s", 1, 64000) is None, "mismatched start must not replace the origin")
    still = store.peek_chunks("s", 1, 0)
    _assert(still.get(0, {}).get("embd") == "keep", "sliding start must leave the origin bucket")
    _assert(full_window_count(8 * 16000, 8 * 16000) == 1, full_window_count(8 * 16000, 8 * 16000))
    _assert(full_window_count(int(7.9 * 16000), 8 * 16000) == 0, "short audio has no full window")
    _assert(full_window_count(20 * 16000, 8 * 16000) == 2, "20s has two full 8s windows")
    src = (SRC / "server.py").read_text(encoding="utf-8")
    _assert("origin_audio" in src, "emit_asr must keep the origin PCM for advance_encoder")
    _assert("not is_final and WS_PARTIAL_WINDOW_SEC" in src, src)
    backend = (SRC / "backend" / "gguf_backend.py").read_text(encoding="utf-8")
    _assert("def advance_encoder" in backend, "shared utt must encode full windows without decode")
    _assert("ensure_origin" in backend, "shared utt must use the origin bucket, not a sliding start")


def test_cache_raw_text() -> None:
    cache = {}
    set_cache_embd(cache, 0, "embd")
    set_cache_raw_text(cache, 0, "百年奋斗")
    _assert(raw_window_text(cache[0]) == "百年奋斗", raw_window_text(cache[0]))
    _assert(should_skip_decode(cache, 0, was_last=False, enabled=True) == (False, None), "raw text is not a freeze")


def test_shared_utt_flag_gates() -> None:
    cfg = (SRC / "config.py").read_text(encoding="utf-8")
    _assert('WS_SHARED_UTT = _safe_bool("WS_SHARED_UTT", "false")' in cfg, "shared utt defaults off")
    _assert('WS_LAZY_PARTIAL = _safe_bool("WS_LAZY_PARTIAL", "false")' in cfg, "lazy partial defaults off")
    _assert('WS_FINAL_YIELD = _safe_bool("WS_FINAL_YIELD", "false")' in cfg, "final yield defaults off")
    _assert("WS_FINAL_YIELD requires WS_SHARED_UTT=true" in cfg, cfg)
    src = (SRC / "server.py").read_text(encoding="utf-8")
    _assert("WS_LAZY_PARTIAL and infer_queue.busy" in src, "lazy partial must wait for a free worker")
    asr = (VENDOR / "qwen_asr_gguf" / "inference" / "asr.py").read_text(encoding="utf-8")
    loop = asr.split("skip_before = max(0, int(skip_decode_before or 0))")[1]
    _assert("if abort_event is not None and abort_event.is_set()" in loop, "encode loop must check abort")
    _assert("yield_event" in loop, "full windows must check yield before encode")
    _assert("skip_decode_before" in asr, "yield/skip path must skip already-raw windows")


def test_skip_discards_stale_overlap() -> None:
    _assert(discard_stale_overlap_after_skip(1), "any silence skip is a new sentence")
    _assert(not discard_stale_overlap_after_skip(0), "no skip keeps hangover overlap")
    # After a pause, previous-sentence stitch would eat 还/那/今.
    _assert(
        strip_boundary("有十分钟，他就到了。", "还有十分钟，他就到了。") == "",
        strip_boundary("有十分钟，他就到了。", "还有十分钟，他就到了。"),
    )
    _assert(
        strip_boundary("我给你打十分。", "那我给你打十分。") == "",
        strip_boundary("我给你打十分。", "那我给你打十分。"),
    )


if __name__ == "__main__":
    test_carry_prefix_tail()
    print("ok test_carry_prefix_tail")
    test_two_tier_silence()
    print("ok test_two_tier_silence")
    test_no_rms_fallback()
    print("ok test_no_rms_fallback")
    test_energy_run_resets_on_silence()
    print("ok test_energy_run_resets_on_silence")
    test_drain_pops_queued_segments()
    print("ok test_drain_pops_queued_segments")
    test_skip_keeps_sustained_energy()
    print("ok test_skip_keeps_sustained_energy")
    test_queue_final_drops_partial()
    print("ok test_queue_final_drops_partial")
    test_hard_cut_rolls_audio_start()
    print("ok test_hard_cut_rolls_audio_start")
    test_hard_cut_silence_selection()
    print("ok test_hard_cut_silence_selection")
    test_drop_before_is_commit_not_last_char()
    print("ok test_drop_before_is_commit_not_last_char")
    test_strip_overlap_prefix()
    print("ok test_strip_overlap_prefix")
    test_boundary_trim_and_stitch()
    print("ok test_boundary_trim_and_stitch")
    test_first_char_boundary_words()
    print("ok test_first_char_boundary_words")
    test_soft_cut_split_not_full_copy()
    print("ok test_soft_cut_split_not_full_copy")
    test_vad_turn_overlap_independent_of_hard_cut()
    print("ok test_vad_turn_overlap_independent_of_hard_cut")
    test_text_ends_sentence()
    print("ok test_text_ends_sentence")
    test_fresh_audio_samples()
    print("ok test_fresh_audio_samples")
    test_hard_cut_holdback_moves_commit()
    print("ok test_hard_cut_holdback_moves_commit")
    test_final_overlap_text_first()
    print("ok test_final_overlap_text_first")
    test_vad_overlap_reties_to_stream()
    print("ok test_vad_overlap_reties_to_stream")
    test_pre_roll_unused_excludes_current_packet()
    print("ok test_pre_roll_unused_excludes_current_packet")
    test_realtime_one_window()
    print("ok test_realtime_one_window")
    test_encoder_cache_and_commit()
    print("ok test_encoder_cache_and_commit")
    test_pending_caption_prefix()
    print("ok test_pending_caption_prefix")
    test_join_partial_tail()
    print("ok test_join_partial_tail")
    test_partial_terminal_punctuation()
    print("ok test_partial_terminal_punctuation")
    test_partial_request_gate()
    print("ok test_partial_request_gate")
    test_ordered_final_buffer()
    print("ok test_ordered_final_buffer")
    test_nonoverlapping_partial_backlog()
    print("ok test_nonoverlapping_partial_backlog")
    test_partial_window_keeps_head_before_first_emit()
    print("ok test_partial_window_keeps_head_before_first_emit")
    test_fallback_empty_final()
    print("ok test_fallback_empty_final")
    test_final_reads_cache_before_drop()
    print("ok test_final_reads_cache_before_drop")
    test_hard_cut_does_not_reset_vad()
    print("ok test_hard_cut_does_not_reset_vad")
    test_skip_discards_stale_overlap()
    print("ok test_skip_discards_stale_overlap")
    test_clocks_and_timing_regex()
    print("ok test_clocks_and_timing_regex")
    test_shared_utt_origin_not_wiped()
    print("ok test_shared_utt_origin_not_wiped")
    test_cache_raw_text()
    print("ok test_cache_raw_text")
    test_shared_utt_flag_gates()
    print("ok test_shared_utt_flag_gates")
    print("ws_tune tests passed")
