"""No-GPU checks for Chinese number ITN and token sync.

    conda activate lingting
    python E2Etest/test_chinese_itn.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from backend.base import WordTimestamp
from chinese_itn import chinese_to_num
import config
from itn import apply_chinese_itn
from token_sync import sync_words_from_text


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_chinese_to_num_cases() -> None:
    _assert(chinese_to_num("十五六个") == "15~16个", chinese_to_num("十五六个"))
    _assert(chinese_to_num("三百块") == "300块", chinese_to_num("三百块"))
    _assert(chinese_to_num("幺九二点幺六八") == "192.168", chinese_to_num("幺九二点幺六八"))
    _assert(chinese_to_num("五点五万亿元") == "5.5万亿元", chinese_to_num("五点五万亿元"))
    _assert(chinese_to_num("一点二万亿元") == "1.2万亿元", chinese_to_num("一点二万亿元"))
    _assert(
        chinese_to_num("国内生产总值达到五点五万亿元") == "国内生产总值达到5.5万亿元",
        chinese_to_num("国内生产总值达到五点五万亿元"),
    )
    _assert(chinese_to_num("百分之百") == "100%", chinese_to_num("百分之百"))
    _assert(chinese_to_num("九月六日") == "9月6日", chinese_to_num("九月六日"))
    _assert(chinese_to_num("九月六号") == "9月6号", chinese_to_num("九月六号"))
    _assert(chinese_to_num("二零二六年九月六号") == "2026年9月6号", chinese_to_num("二零二六年九月六号"))
    _assert(chinese_to_num("十一点五十") == "11:50", chinese_to_num("十一点五十"))
    _assert(chinese_to_num("十一点五") == "11.5", chinese_to_num("十一点五"))
    _assert(chinese_to_num("零点零五") == "0.05", chinese_to_num("零点零五"))
    _assert(chinese_to_num("十二点零五") == "12.05", chinese_to_num("十二点零五"))
    _assert(
        chinese_to_num("二零二六年九月六日上午十一点五十") == "2026年9月6日上午11:50",
        chinese_to_num("二零二六年九月六日上午十一点五十"),
    )
    _assert(
        chinese_to_num("2026年9月6号上午11点50分10秒") == "2026年9月6号上午11:50:10",
        chinese_to_num("2026年9月6号上午11点50分10秒"),
    )
    _assert(chinese_to_num("上午11点50") == "上午11:50", chinese_to_num("上午11点50"))
    _assert(chinese_to_num("11点50") == "11:50", chinese_to_num("11点50"))
    _assert(chinese_to_num("下午3点5") == "下午3:05", chinese_to_num("下午3点5"))
    _assert(
        chinese_to_num("在国内外形势十分复杂") == "在国内外形势十分复杂",
        chinese_to_num("在国内外形势十分复杂"),
    )
    _assert(chinese_to_num("今天我给你打十分") == "今天我给你打10分", chinese_to_num("今天我给你打十分"))
    _assert(chinese_to_num("十分钟后出发") == "10分钟后出发", chinese_to_num("十分钟后出发"))
    _assert(chinese_to_num("打二十分") == "打20分", chinese_to_num("打二十分"))


def test_sync_words_span() -> None:
    words = [
        WordTimestamp("十", 1.00, 1.10),
        WordTimestamp("五", 1.10, 1.20),
        WordTimestamp("个", 1.20, 1.30),
    ]
    out = sync_words_from_text(words, "15个")
    _assert("".join(w.word for w in out) == "15个", [w.word for w in out])
    fifteen = next(w for w in out if w.word == "15")
    _assert(abs(fifteen.start - 1.00) < 1e-6, fifteen)
    _assert(abs(fifteen.end - 1.20) < 1e-6, fifteen)
    ge = next(w for w in out if w.word == "个")
    _assert(abs(ge.start - 1.20) < 1e-6, ge)


def test_apply_respects_flag() -> None:
    words = [WordTimestamp("三", 0.0, 0.1), WordTimestamp("百", 0.1, 0.2)]
    prev = config.FORMAT_NUM
    try:
        config.FORMAT_NUM = True
        text, synced = apply_chinese_itn("三百", words)
        _assert(text == "300", text)
        _assert(synced[0].word == "300", synced)
        config.FORMAT_NUM = False
        text, synced = apply_chinese_itn("三百", words)
        _assert(text == "三百", text)
        _assert([w.word for w in synced] == ["三", "百"], synced)
    finally:
        config.FORMAT_NUM = prev


def test_server_carry_uses_raw() -> None:
    src = (Path(__file__).resolve().parents[1] / "src" / "server.py").read_text(encoding="utf-8")
    _assert("raw_for_carry = text" in src, "final must snapshot text before ITN")
    _assert("carry_prefix_tail(carry_src, 16)" in src, src)
    _assert("carry_src = raw_for_carry or text" in src, "carry must prefer pre-ITN text")
    final_fn = src.split("if leftover_echo(stitch_src, text):")[1].split("else:")[0]
    _assert("apply_chinese_itn(" in final_fn, "ITN must run on WS final")
    partial_fn = src.split("else:")[-1].split("payload = {")[0]
    _assert("apply_chinese_itn(" not in partial_fn, "partial must not run ITN")
    offline = (Path(__file__).resolve().parents[1] / "src" / "offline_tasks.py").read_text(encoding="utf-8")
    _assert("apply_chinese_itn(" in offline, "offline JSON must run ITN")


def main() -> int:
    tests = [
        test_chinese_to_num_cases,
        test_sync_words_span,
        test_apply_respects_flag,
        test_server_carry_uses_raw,
    ]
    for fn in tests:
        fn()
        print("ok", fn.__name__)
    print("chinese_itn tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
