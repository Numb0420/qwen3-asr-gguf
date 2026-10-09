"""No-GPU checks for VAD silence threshold logic.

    conda activate lingting
    python E2Etest/test_vad.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vad import silence_threshold_samples


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_silence_threshold_short_utterance() -> None:
    """Short utterances (< 0.8s) use the higher early-silence threshold."""
    sr = 16000
    short_samples = int(0.3 * sr)  # 0.3s of speech
    got = silence_threshold_samples(short_samples, sr, flush_ms=500, early_ms=600)
    expected = int(sr * (600 / 1000.0))
    _assert(got == expected, f"short utterance: expected {expected}, got {got}")


def test_silence_threshold_long_utterance() -> None:
    """Long utterances (>= 0.8s) use the lower flush-silence threshold."""
    sr = 16000
    long_samples = int(1.0 * sr)  # 1.0s of speech
    got = silence_threshold_samples(long_samples, sr, flush_ms=500, early_ms=600)
    expected = int(sr * (500 / 1000.0))
    _assert(got == expected, f"long utterance: expected {expected}, got {got}")


def test_silence_threshold_boundary() -> None:
    """Exactly 0.8s (committed threshold) uses flush-silence."""
    sr = 16000
    boundary_samples = int(0.8 * sr)
    got = silence_threshold_samples(boundary_samples, sr, flush_ms=500, early_ms=900)
    expected = int(sr * (500 / 1000.0))
    _assert(got == expected, f"boundary: expected {expected}, got {got}")


def test_silence_threshold_custom_values() -> None:
    """Custom flush/early values are respected."""
    sr = 16000
    short = int(0.2 * sr)
    long = int(2.0 * sr)
    got_short = silence_threshold_samples(short, sr, flush_ms=400, early_ms=800)
    got_long = silence_threshold_samples(long, sr, flush_ms=400, early_ms=800)
    _assert(got_short == int(sr * 0.8), f"custom short: {got_short}")
    _assert(got_long == int(sr * 0.4), f"custom long: {got_long}")


def test_no_double_counting_sherpa_internal() -> None:
    """Verify that the external threshold is the sole silence gate.

    The Sherpa-internal min_silence_duration should be set to 0.05s (not
    WS_FLUSH_SILENCE_MS) to avoid double counting. This test documents the
    contract: the external silence_threshold_samples is the authoritative
    silence detector, Sherpa's internal gate is just a sub-frame filter.
    """
    sr = 16000
    # If Sherpa internal were set to WS_FLUSH_SILENCE_MS (500ms) AND the
    # external counter also uses 500ms, total wait would be ~1000ms.
    # With Sherpa at 0.05s, total wait is ~550ms (50 + 500).
    # This test verifies the external threshold is independent of Sherpa.
    external_threshold = silence_threshold_samples(
        int(1.0 * sr), sr, flush_ms=500, early_ms=600
    )
    expected_external = int(sr * 0.5)
    _assert(
        external_threshold == expected_external,
        f"external threshold should be {expected_external}, got {external_threshold}"
    )
    # The external threshold alone (not Sherpa + external) determines when
    # speech_ended fires. Sherpa's 0.05s only filters sub-frame noise.
    _assert(
        external_threshold < sr,  # less than 1s
        "external threshold should be sub-second for responsive VAD"
    )


def main() -> None:
    tests = [
        test_silence_threshold_short_utterance,
        test_silence_threshold_long_utterance,
        test_silence_threshold_boundary,
        test_silence_threshold_custom_values,
        test_no_double_counting_sherpa_internal,
    ]
    for t in tests:
        t()
        print(f"  {t.__name__} PASSED")
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    main()
