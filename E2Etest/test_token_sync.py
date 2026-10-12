"""CPU checks for complete ITN numbers and their exported timestamps."""
import sys
from pathlib import Path
import unittest
import ast
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from backend.base import WordTimestamp
from token_sync import sync_words_from_text, sync_words_from_replacements
from segments import words_to_segments


class NumberSyncTests(unittest.TestCase):
    def test_decimal_from_individual_chinese_tokens(self):
        raw = "五十六点五的嘛？"
        words = [WordTimestamp(ch, 10 + i / 10, 10 + (i + 1) / 10)
                 for i, ch in enumerate(raw)]
        out = sync_words_from_text(words, "56.5的嘛？")
        self.assertEqual(out[0], WordTimestamp("56.5", 10.0, 10.5))
        segment = words_to_segments(out)[0]
        self.assertEqual(segment["chars"][0],
                         {"text": "56.5", "start": 10.0, "end": 10.5})
        self.assertEqual("".join(c["text"] for c in segment["chars"]), "56.5的嘛？")
        self.assertEqual(segment["chars"][-1]["start"], out[-2].end)

    def test_decimal_from_grouped_source_token(self):
        out = sync_words_from_text([WordTimestamp("五十六点五", 102.816, 103.376)], "56.5")
        self.assertEqual(out, [WordTimestamp("56.5", 102.816, 103.376)])

    def test_partial_conversion_retains_full_number_span(self):
        out = sync_words_from_text([WordTimestamp("56", 1.0, 1.3),
                                    WordTimestamp("点", 1.3, 1.4),
                                    WordTimestamp("五", 1.4, 1.5)], "56.5")
        self.assertEqual(out, [WordTimestamp("56.5", 1.0, 1.5)])

    def test_existing_overlapping_number_tokens_are_merged(self):
        out = sync_words_from_text([WordTimestamp("56", 102.816, 103.376),
                                    WordTimestamp(".", 103.376, 103.376),
                                    WordTimestamp("5", 102.816, 103.376)], "56.5")
        self.assertEqual(out, [WordTimestamp("56.5", 102.816, 103.376)])

    def test_sentence_period_and_separate_numbers_stay_separate(self):
        raw = "五十六点五。六点五万，三。"
        words = [WordTimestamp(ch, i / 10, (i + 1) / 10) for i, ch in enumerate(raw)]
        out = sync_words_from_text(words, "56.5。6.5万，3。")
        self.assertEqual([w.word for w in out], ["56.5", "。", "6.5", "万", "，", "3", "。"])
        self.assertEqual([s["text"] for s in words_to_segments(out)], ["56.5。", "6.5万，3。"])


class HotwordSyncTests(unittest.TestCase):
    def test_changed_length_uses_entire_matched_span(self):
        out = sync_words_from_replacements(
            [WordTimestamp("错", 1.0, 1.2), WordTimestamp("词", 1.2, 1.4)],
            "错词", [(0, 2, "正确热词")])
        self.assertEqual(out, [WordTimestamp("正确热词", 1.0, 1.4)])

    def test_repeated_word_uses_selected_occurrence(self):
        words = [WordTimestamp(ch, i, i + 0.5) for i, ch in enumerate("错词，错词。")]
        out = sync_words_from_replacements(words, "错词，错词。", [(3, 5, "正确热词")])
        self.assertEqual(out[:3], words[:3])
        self.assertEqual(out[3], WordTimestamp("正确热词", 3, 4.5))
        self.assertEqual(out[4], words[5])

    def test_multiple_replacements_keep_symbols_and_decimal(self):
        raw = "千问ASR，拉马点CPP，56.5。"
        words = [WordTimestamp(ch, i / 10, (i + 1) / 10) for i, ch in enumerate(raw[:13])]
        words += [WordTimestamp("56.5", 2.0, 2.6), WordTimestamp("。", 2.6, 2.6)]
        self.assertEqual("".join(w.word for w in words), raw)
        out = sync_words_from_replacements(words, raw,
                                          [(6, 12, "Llama.cpp"), (0, 5, "Qwen3-ASR")])
        self.assertEqual([w.word for w in out], ["Qwen3-ASR", "，", "Llama.cpp", "，", "56.5", "。"])
        segment = words_to_segments(out)[0]
        self.assertEqual([c["text"] for c in segment["chars"]],
                         ["Qwen3-ASR", "，", "Llama.cpp", "，", "56.5", "。"])
        self.assertEqual(out[0], WordTimestamp("Qwen3-ASR", 0.0, 0.5))
        self.assertEqual(out[2], WordTimestamp("Llama.cpp", 0.6, 1.2))

    def test_match_inside_existing_token_preserves_surroundings(self):
        out = sync_words_from_replacements([WordTimestamp("甲错词乙", 1, 2)],
                                          "甲错词乙", [(1, 3, "正确热词")])
        self.assertEqual([w.word for w in out], ["甲", "正确热词", "乙"])
        self.assertEqual((out[1].start, out[1].end), (1, 2))

    def test_no_alignment_and_invalid_spans(self):
        self.assertEqual(sync_words_from_replacements(None, "错词", [(0, 2, "热词")]), [])
        with self.assertRaises(ValueError):
            sync_words_from_replacements([WordTimestamp("错词", 1, 2)],
                                         "错词", [(0, 2, "热词"), (1, 2, "词")])

    def test_actual_resolver_retains_original_positions(self):
        # Exercise the actual conflict resolver without loading phoneme runtimes.
        path = Path(__file__).resolve().parents[1] / "src/hotword/hot_phoneme.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PhonemeCorrector")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_resolve_and_replace")
        namespace = {}
        module = ast.Module(body=[ast.ImportFrom(module="__future__",
                            names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
        corrector = SimpleNamespace(threshold=0.85, _tokenize_semantic_words=lambda _: [],
                                    _is_blacklisted=lambda *args: False)
        match = SimpleNamespace(start=3, end=5, hotword="正确热词", score=0.99)
        result = namespace["_resolve_and_replace"](corrector, "错词，错词。", [match], 5)
        self.assertEqual(result[0], "错词，正确热词。")
        self.assertEqual(result[3], [match])

    def test_apply_hotword_uses_match_span(self):
        path = Path(__file__).resolve().parents[1] / "src/hotword_correct.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        method = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "apply_hotword")
        result = SimpleNamespace(text="正确热词。", matches=[("错词", "正确热词", 0.99)],
                                 replacements=[SimpleNamespace(start=0, end=2, hotword="正确热词")])
        namespace = dict(config=SimpleNamespace(HOTWORD_ENABLED=True), WordTimestamp=WordTimestamp,
                         _get_corrector=lambda: SimpleNamespace(correct=lambda _: result),
                         log=SimpleNamespace(info=lambda *args: None, warning=lambda *args: None),
                         sync_words_from_replacements=sync_words_from_replacements)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
        text, words = namespace["apply_hotword"]("错词。", [WordTimestamp("错", 1, 1.2),
                           WordTimestamp("词", 1.2, 1.4), WordTimestamp("。", 1.4, 1.4)])
        self.assertEqual(text, "正确热词。")
        self.assertEqual(words[0], WordTimestamp("正确热词", 1, 1.4))


if __name__ == "__main__":
    unittest.main()
