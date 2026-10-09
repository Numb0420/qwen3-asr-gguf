"""User-visible realtime and offline disfluency cleanup cases."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fillers import clean_fillers, clean_filler_segments


def test_meeting_fillers_and_empty_turns():
    assert clean_fillers("呃，") == ""
    assert clean_fillers("啊，") == ""
    assert clean_fillers("嗯嗯，") == ""
    assert clean_fillers("但是呃我相信") == "但是我相信"
    assert clean_fillers("进行非常嗯深度 Coding 的人") == "进行非常深度 Coding 的人"
    assert clean_fillers("感觉这个框架给我自己带来，哎，更多是一个") == "感觉这个框架给我自己带来，更多是一个"


def test_meaningful_interjections_and_terms_survive():
    assert clean_fillers("好啊，我们继续") == "好啊，我们继续"
    assert clean_fillers("哎呀，这个不错") == "哎呀，这个不错"
    assert clean_fillers("嗯哼，继续") == "嗯哼，继续"
    assert clean_fillers("哎哟，怎么了") == "哎哟，怎么了"
    assert clean_fillers("呃逆的治疗") == "呃逆的治疗"
    assert clean_fillers("Agent 和 Coding") == "Agent 和 Coding"


def test_offline_segments_keep_matching_chars_and_times():
    segments = [
        {"index": 1, "start": 0.0, "end": 0.5, "text": "呃。", "punctuation": "。", "speaker": None,
         "chars": [{"text": "呃", "start": 0.1, "end": 0.4}]},
        {"index": 2, "start": 0.5, "end": 2.0, "text": "但是呃我相信。", "punctuation": "。", "speaker": None,
         "chars": [
             {"text": ch, "start": 0.5 + i * 0.2, "end": 0.7 + i * 0.2}
             for i, ch in enumerate("但是呃我相信")
         ]},
    ]
    result = clean_filler_segments(segments)
    assert len(result) == 1
    assert result[0]["index"] == 1
    assert result[0]["text"] == "但是我相信。"
    assert "".join(c["text"] for c in result[0]["chars"]) == "但是我相信"
    assert result[0]["start"] == 0.5
    assert result[0]["end"] == segments[1]["chars"][-1]["end"]
