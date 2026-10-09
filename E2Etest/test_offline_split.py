"""No-GPU checks for offline silence chunk planning.

    conda activate lingting
    python E2Etest/test_offline_split.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from backend.base import WordTimestamp
from offline_split import (
    SplitConfig,
    drop_overlap_tokens,
    plan_chunks,
    shift_words,
    stitch_chunk_text,
    stitch_chunk_words,
    strip_forced_cut_words,
)
from offline_stitch import (
    append_offline_chunk,
    drop_collapsed_cut_tokens,
    finalize_offline_chunk,
)
from ws_overlap import strip_forced_cut_stop


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_short_file_one_chunk() -> None:
    cfg = SplitConfig()
    plans = plan_chunks(11.0, [], cfg)
    _assert(len(plans) == 1, f"expected 1 chunk, got {plans}")
    _assert(plans[0].audio_start == 0.0, str(plans[0]))
    _assert(plans[0].audio_end == 11.0, str(plans[0]))
    _assert(plans[0].kind == "tail", str(plans[0]))


def test_silence_near_target() -> None:
    cfg = SplitConfig()
    silences = [(34.0, 35.2)]
    plans = plan_chunks(80.0, silences, cfg)
    _assert(len(plans) >= 2, str(plans))
    _assert(plans[0].kind == "silence", str(plans[0]))
    _assert(abs(plans[0].audio_end - 34.6) < 0.01, str(plans[0]))
    _assert(plans[0].drop_before is None, str(plans[0]))
    _assert(abs(plans[1].audio_start - (plans[0].audio_end - 0.4)) < 0.01, str(plans))
    _assert(plans[1].drop_before == plans[0].audio_end, str(plans[1]))


def test_silence_overlap_when_under_max() -> None:
    cfg = SplitConfig()
    silences = [(34.0, 35.2), (69.0, 70.2)]
    plans = plan_chunks(100.0, silences, cfg)
    _assert(plans[0].kind == "silence", str(plans[0]))
    _assert(plans[1].kind == "silence", str(plans[1]))
    _assert(abs(plans[1].audio_start - (plans[0].audio_end - 0.4)) < 0.02, str(plans))
    _assert(plans[1].drop_before == plans[0].audio_end, str(plans[1]))
    _assert(plans[1].duration <= cfg.max_chunk + 1e-6, str(plans[1]))


def test_hard_cut_after_silence_stays_max() -> None:
    cfg = SplitConfig()
    silences = [(34.0, 35.2)]
    plans = plan_chunks(134.0, silences, cfg)
    _assert(plans[0].kind == "silence", str(plans[0]))
    hard = next(p for p in plans if p.kind == "hard")
    _assert(hard.kind == "hard", str(hard))
    _assert(hard.duration <= cfg.max_chunk + cfg.silence_overlap + 1e-6, str(hard))


def test_hard_cut_overlap() -> None:
    cfg = SplitConfig()
    plans = plan_chunks(80.0, [], cfg)
    _assert(plans[0].kind == "hard", str(plans[0]))
    _assert(abs(plans[0].audio_end - 40.0) < 1e-6, str(plans[0]))
    _assert(abs(plans[1].audio_start - 38.5) < 1e-6, str(plans[1]))
    _assert(plans[1].drop_before == 40.0, str(plans[1]))


def test_no_overlap_without_aligner() -> None:
    cfg = SplitConfig(allow_hard_overlap=False)
    plans = plan_chunks(80.0, [], cfg)
    _assert(plans[1].audio_start == 40.0, str(plans[1]))
    _assert(plans[1].drop_before == 40.0, str(plans[1]))
    silences = [(34.0, 35.2)]
    plans = plan_chunks(80.0, silences, cfg)
    _assert(plans[1].audio_start == plans[0].audio_end, str(plans))
    _assert(plans[1].drop_before is None, str(plans[1]))


def test_prefer_near_target_over_long_silence() -> None:
    cfg = SplitConfig()
    silences = [(31.5, 31.9), (34.2, 34.9), (38.1, 39.5)]
    plans = plan_chunks(80.0, silences, cfg)
    _assert(plans[0].kind == "silence", str(plans[0]))
    # 34.2-34.9 mid=34.55 is closest to 35 among >=400ms
    _assert(abs(plans[0].audio_end - 34.55) < 0.02, str(plans[0]))


def test_drop_overlap_tokens() -> None:
    words = [
        WordTimestamp("某", 38.7, 38.9),
        WordTimestamp("项", 39.12, 39.28),
        WordTimestamp("后", 40.05, 40.20),
    ]
    kept = drop_overlap_tokens(words, cut=40.0, eps=0.15)
    _assert([w.word for w in kept] == ["后"], kept)


def test_shift_words() -> None:
    words = [WordTimestamp("项", 0.62, 0.78)]
    shifted = shift_words(words, 38.5)
    _assert(shifted[0].start == 39.12, shifted)
    _assert(shifted[0].end == 39.28, shifted)


def test_stitch_chunk_text() -> None:
    _assert(stitch_chunk_text("实现。", "现国家富强") == "实现国家富强", stitch_chunk_text("实现。", "现国家富强"))
    _assert(stitch_chunk_text("文化自。", "自信，做到") == "文化自信，做到", stitch_chunk_text("文化自。", "自信，做到"))
    _assert(stitch_chunk_text("今天天气很好。", "我们去公园。") == "今天天气很好。我们去公园。", stitch_chunk_text("今天天气很好。", "我们去公园。"))
    _assert(
        stitch_chunk_text("发展中国特色社会主义。", "的需要是增强政治意识") == "发展中国特色社会主义的需要是增强政治意识",
        stitch_chunk_text("发展中国特色社会主义。", "的需要是增强政治意识"),
    )
    _assert(
        stitch_chunk_text("实现中华民族伟大复兴。", "的中国梦而继续奋斗的需要") == "实现中华民族伟大复兴的中国梦而继续奋斗的需要",
        stitch_chunk_text("实现中华民族伟大复兴。", "的中国梦而继续奋斗的需要"),
    )
    _assert(strip_forced_cut_stop("始终坚持。") == "始终坚持", strip_forced_cut_stop("始终坚持。"))
    _assert(
        stitch_chunk_text("始终坚持", "共产主义理想和社会主义信念") == "始终坚持共产主义理想和社会主义信念",
        stitch_chunk_text("始终坚持", "共产主义理想和社会主义信念"),
    )


def test_stitch_chunk_words() -> None:
    prev = [
        WordTimestamp("实", 33.8, 34.0),
        WordTimestamp("现", 34.0, 34.15),
        WordTimestamp("。", 34.15, 34.15),
    ]
    nxt = [
        WordTimestamp("现", 34.2, 34.35),
        WordTimestamp("国", 34.35, 34.5),
    ]
    out = stitch_chunk_words(prev, nxt)
    _assert([w.word for w in out] == ["实", "现", "国"], [w.word for w in out])
    real = [
        WordTimestamp("好", 34.0, 34.2),
        WordTimestamp("。", 34.2, 34.2),
    ]
    nxt2 = [WordTimestamp("我", 34.9, 35.1)]
    out2 = stitch_chunk_words(real, nxt2)
    _assert([w.word for w in out2] == ["好", "。", "我"], [w.word for w in out2])
    de = [
        WordTimestamp("义", 39.8, 40.0),
        WordTimestamp("。", 40.0, 40.0),
    ]
    de_n = [WordTimestamp("的", 40.1, 40.2), WordTimestamp("需", 40.2, 40.4)]
    out3 = stitch_chunk_words(de, de_n)
    _assert([w.word for w in out3] == ["义", "的", "需"], [w.word for w in out3])
    _assert([w.word for w in strip_forced_cut_words(de)] == ["义"], strip_forced_cut_words(de))


def test_offline_hard_cut_uses_stop_strip() -> None:
    src = (Path(__file__).resolve().parents[1] / "src" / "offline_tasks.py").read_text(encoding="utf-8")
    _assert("finalize_offline_chunk(" in src, "offline must reuse WS boundary stitch")
    _assert("append_offline_chunk(" in src, "offline must keep 1-char word stitch")
    _assert("kind=plan.kind" in src, src)


def _words(text: str, start: float, step: float = 0.08) -> list[WordTimestamp]:
    out: list[WordTimestamp] = []
    t = start
    for ch in text:
        out.append(WordTimestamp(ch, round(t, 3), round(t + step, 3)))
        t += step
    return out


def _collapsed(text: str, at: float) -> list[WordTimestamp]:
    return [WordTimestamp(ch, at, at) for ch in text]


def test_offline_strip_boundary_party() -> None:
    prev = "坚决维护党中央权威和"
    nxt = "党中央权威和集中统一领导"
    text, words = finalize_offline_chunk(
        nxt,
        _words(nxt, 80.0),
        audio_start=0.0,
        audio_end=120.0,
        drop_before=80.0,
        kind="hard",
        prev_text=prev,
        prev_words=_words(prev, 77.0),
    )
    _assert(text == "集中统一领导", text)
    _assert("".join(w.word for w in words) == "集中统一领导", words)


def test_offline_strip_打牺牲() -> None:
    prev = "付出巨大牺牲积累的一系列"
    nxt = "打牺牲积累的一系列独创性经验"
    text, words = finalize_offline_chunk(
        nxt,
        _words(nxt, 160.0),
        audio_start=0.0,
        audio_end=200.0,
        drop_before=160.0,
        kind="hard",
        prev_text=prev,
        prev_words=_words(prev, 157.0),
    )
    _assert(text == "独创性经验", text)
    _assert("".join(w.word for w in words) == "独创性经验", words)


def test_offline_collapsed_cut_tokens() -> None:
    prev = "比去年同期"
    words = _collapsed("万亿元，比去年同期", 280.02) + _words("增长", 280.18)
    text, kept = finalize_offline_chunk(
        "万亿元，比去年同期增长",
        words,
        audio_start=0.0,
        audio_end=320.0,
        drop_before=280.0,
        kind="hard",
        prev_text=prev,
        prev_words=_words(prev, 279.3, 0.12),
    )
    _assert(text == "增长", text)
    _assert("".join(w.word for w in kept) == "增长", kept)
    joined = stitch_chunk_text(prev, text)
    _assert(joined == "比去年同期增长", joined)


def test_offline_collapsed_这项新规() -> None:
    prev = "这项新规"
    words = _collapsed("费成本。这项新规", 319.86) + _words("抹平了借贷双方的信息差。", 320.1)
    text, kept = finalize_offline_chunk(
        "费成本。这项新规抹平了借贷双方的信息差。",
        words,
        audio_start=0.0,
        audio_end=360.0,
        drop_before=320.0,
        kind="hard",
        prev_text=prev,
        prev_words=_words(prev, 319.14, 0.16),
    )
    _assert(text.startswith("抹平了"), text)
    _assert("这项新规" not in text, text)
    _assert("费成本" not in "".join(w.word for w in kept), kept)


def test_offline_intra_chunk_collapsed_tail() -> None:
    head = _words("坚决维护", 77.4)
    extra = _collapsed("党中央权威和", 78.5)
    real = _words("党中央权威和", 78.74)
    text, kept = finalize_offline_chunk(
        "坚决维护党中央权威和。",
        head + extra + real,
        audio_start=0.0,
        audio_end=80.0,
        drop_before=40.0,
        kind="hard",
    )
    _assert(text == "坚决维护党中央权威和", text)
    _assert("".join(w.word for w in kept) == "坚决维护党中央权威和", [w.word for w in kept])


def test_offline_leading_short_after_gap() -> None:
    prev = "中国发展从此开启了新纪元"
    nxt = "新晋。3分钟了解国家大事。"
    words = _words("新晋。", 266.5) + _words("3分钟了解国家大事。", 266.7)
    text, kept = finalize_offline_chunk(
        nxt,
        words,
        audio_start=0.0,
        audio_end=280.0,
        drop_before=240.0,
        kind="hard",
        prev_text=prev,
        prev_words=_words(prev, 237.0),
    )
    _assert(text.startswith("3分钟"), text)
    _assert("新晋" not in text, text)
    _assert("".join(w.word for w in kept).startswith("3分钟"), kept)


def test_offline_keep_real_sentences() -> None:
    prev = "今天天气很好。"
    nxt = "我们去公园。"
    text, kept = finalize_offline_chunk(
        nxt,
        _words(nxt, 34.9),
        audio_start=0.0,
        audio_end=40.0,
        drop_before=None,
        kind="silence",
        prev_text=prev,
        prev_words=_words(prev, 33.0),
    )
    _assert(text == "我们去公园。", text)
    combined = stitch_chunk_text(prev, text)
    _assert(combined == "今天天气很好。我们去公园。", combined)
    _assert([w.word for w in kept][:2] == ["我", "们"], kept)


def test_offline_de_after_false_period() -> None:
    prev = "发展中国特色社会主义"
    nxt = "的需要是增强政治意识"
    text, kept = finalize_offline_chunk(
        nxt,
        _words(nxt, 40.1),
        audio_start=0.0,
        audio_end=80.0,
        drop_before=40.0,
        kind="hard",
        prev_text=prev,
        prev_words=_words(prev, 36.0),
    )
    _assert(text == "的需要是增强政治意识", text)
    joined = stitch_chunk_text(prev + "。", text)
    _assert(joined == "发展中国特色社会主义的需要是增强政治意识", joined)


def test_offline_leftover_echo_skipped() -> None:
    text, kept = finalize_offline_chunk(
        "中国。",
        _words("中国。", 40.1),
        audio_start=0.0,
        audio_end=80.0,
        drop_before=40.0,
        kind="hard",
        prev_text="才能发展中国。",
        prev_words=_words("才能发展中国。", 38.0),
    )
    _assert(text == "", text)
    _assert(kept == [], kept)


def test_drop_collapsed_keeps_mid_sentence() -> None:
    words = [WordTimestamp("的", 50.0, 50.0), WordTimestamp("需", 50.1, 50.3)]
    kept = drop_collapsed_cut_tokens(words, cut=80.0, window=1.5)
    _assert([w.word for w in kept] == ["的", "需"], kept)
    _assert(append_offline_chunk([], kept) == kept, kept)


def test_offline_keeps_一系列_and_五点一() -> None:
    prev = "付出巨大牺牲积累"
    words = _words("的", 159.78) + _collapsed("一", 159.86) + _words("系列独创性经验", 159.86)
    text, kept = finalize_offline_chunk(
        "的一系列独创性经验",
        words,
        audio_start=0.0,
        audio_end=160.0,
        drop_before=120.0,
        kind="hard",
        prev_text=prev,
        prev_words=_words(prev, 157.0),
    )
    _assert("一系列" in text, text)
    _assert("".join(w.word for w in kept).count("一") >= 1, kept)

    prev2 = "比去年同期"
    num = _collapsed("五", 280.02) + _words("点一个百分点", 280.66)
    text2, kept2 = finalize_offline_chunk(
        "增长五点一个百分点",
        _words("增长", 280.18) + num,
        audio_start=0.0,
        audio_end=320.0,
        drop_before=280.0,
        kind="hard",
        prev_text=prev2,
        prev_words=_words(prev2, 279.3, 0.12),
    )
    _assert("五" in "".join(w.word for w in kept2), kept2)
    _assert("0.1" not in text2, text2)


def test_offline_keeps_正确革命道路() -> None:
    prev = "武装夺取政权的正"
    nxt = "的正确革命道路，创立了"
    text, words = finalize_offline_chunk(
        nxt,
        _words(nxt, 167.3),
        audio_start=0.0,
        audio_end=200.0,
        drop_before=160.0,
        kind="hard",
        prev_text=prev,
        prev_words=_words(prev, 165.0),
    )
    _assert("革命道路" in text, text)
    _assert("正确革命道路" in prev + text or "确革命道路" in text, text)


def main() -> int:
    tests = [
        test_short_file_one_chunk,
        test_silence_near_target,
        test_silence_overlap_when_under_max,
        test_hard_cut_overlap,
        test_hard_cut_after_silence_stays_max,
        test_no_overlap_without_aligner,
        test_prefer_near_target_over_long_silence,
        test_drop_overlap_tokens,
        test_shift_words,
        test_stitch_chunk_text,
        test_stitch_chunk_words,
        test_offline_hard_cut_uses_stop_strip,
        test_offline_strip_boundary_party,
        test_offline_strip_打牺牲,
        test_offline_collapsed_cut_tokens,
        test_offline_collapsed_这项新规,
        test_offline_intra_chunk_collapsed_tail,
        test_offline_leading_short_after_gap,
        test_offline_keep_real_sentences,
        test_offline_de_after_false_period,
        test_offline_leftover_echo_skipped,
        test_drop_collapsed_keeps_mid_sentence,
        test_offline_keeps_一系列_and_五点一,
        test_offline_keeps_正确革命道路,
    ]
    for fn in tests:
        fn()
        print("ok", fn.__name__)
    print("offline_split tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
