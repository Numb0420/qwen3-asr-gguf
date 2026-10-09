from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import (
    TARGET_SR,
    VAD_MODEL_PATH,
    WS_COMMITTED_SPEECH_SECONDS,
    WS_EARLY_SILENCE_MS,
    WS_FLUSH_SILENCE_MS,
)
from logger import log


@dataclass
class VadEvent:
    is_speech: bool
    speech_ended: bool
    # Current continuous silence in milliseconds (updated every frame).
    silence_ms: float = 0.0
    # True on the frame where speech resumes after a silence gap shorter than vad_end.
    speech_resumed: bool = False
    # Silence milliseconds immediately before this resume (0.0 if no resume).
    resumed_after_silence_ms: float = 0.0


def silence_threshold_samples(
    speech_samples: int,
    sample_rate: int,
    flush_ms: int = WS_FLUSH_SILENCE_MS,
    early_ms: int = WS_EARLY_SILENCE_MS,
    committed_s: float = WS_COMMITTED_SPEECH_SECONDS,
) -> int:
    """Longer silence is required before the utterance has ~0.8s of speech."""
    committed = int(committed_s * sample_rate)
    ms = flush_ms if speech_samples >= committed else early_ms
    return int(sample_rate * (ms / 1000.0))


# Skip-path only. RealtimeVad.accept_audio must not use this to invent speech_ended.
ENERGY_ONSET_RMS = 0.01
ENERGY_ONSET_MS = 200


def chunk_rms(pcm: np.ndarray) -> float:
    samples = np.asarray(pcm, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples * samples)))


def advance_energy_run(run_samples: int, pcm: np.ndarray, *, threshold: float = ENERGY_ONSET_RMS) -> int:
    """Accumulate samples while the packet stays above the RMS gate. Silence resets."""
    samples = np.asarray(pcm, dtype=np.float32).reshape(-1)
    if samples.size == 0 or chunk_rms(samples) < threshold:
        return 0
    return int(run_samples) + int(samples.size)


def energy_onset_ready(run_samples: int, sample_rate: int, onset_ms: int = ENERGY_ONSET_MS) -> bool:
    return int(run_samples) >= int(sample_rate * (onset_ms / 1000.0))


class RealtimeVad:
    """Streaming Silero VAD via sherpa-onnx. One instance per WS session."""

    def __init__(self, model_path: Path | None = None, sample_rate: int = TARGET_SR):
        self.sample_rate = sample_rate
        self.model_path = Path(model_path or VAD_MODEL_PATH)
        self._vad = None
        self._window = 512
        self._pending = np.zeros(0, dtype=np.float32)
        self._enabled = False
        self._speech = False
        self._speech_samples = 0
        self._silence_samples = 0
        self._init()

    def _init(self) -> None:
        if not self.model_path.exists():
            log.warning("VAD model missing at {}, falling back to always-speech", self.model_path)
            return
        try:
            import sherpa_onnx

            cfg = sherpa_onnx.VadModelConfig()
            cfg.silero_vad.model = str(self.model_path)
            cfg.silero_vad.threshold = 0.5
            # Sherpa internal silence gate — set very low so it only filters
            # sub-frame noise. The real silence threshold is enforced by the
            # external silence_samples counter (WS_FLUSH_SILENCE_MS /
            # WS_EARLY_SILENCE_MS). Setting this to WS_FLUSH_SILENCE_MS would
            # cause double counting: Sherpa keeps is_speech_detected() true
            # during its internal wait, then the external counter adds another
            # full threshold on top.
            cfg.silero_vad.min_silence_duration = 0.05
            cfg.silero_vad.min_speech_duration = 0.15
            cfg.sample_rate = self.sample_rate
            self._vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=60)
            window = getattr(cfg.silero_vad, "window_size", None)
            if window:
                self._window = int(window)
            self._enabled = True
            log.info("RealtimeVad ready | model={}", self.model_path)
        except Exception as e:
            log.error("RealtimeVad init failed: {} — fallback always-speech", e)
            self._vad = None
            self._enabled = False

    def _drain_queued_segments(self) -> None:
        """Pop finished sherpa segments. A full queue stops is_speech_detected()."""
        vad = self._vad
        if vad is None:
            return
        empty = getattr(vad, "empty", None)
        pop = getattr(vad, "pop", None)
        if not callable(pop):
            return
        try:
            guard = 0
            while True:
                if callable(empty):
                    done = bool(empty())
                elif empty is None:
                    done = True
                else:
                    done = bool(empty)
                if done or guard >= 1000:
                    return
                pop()
                guard += 1
        except Exception:
            return

    def accept_audio(self, pcm: np.ndarray) -> VadEvent:
        samples = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return VadEvent(is_speech=self._speech, speech_ended=False)

        if not self._enabled or self._vad is None:
            self._speech = True
            return VadEvent(is_speech=True, speech_ended=False)

        self._pending = np.concatenate([self._pending, samples])
        speech_this = False
        while self._pending.size >= self._window:
            chunk = self._pending[: self._window]
            self._pending = self._pending[self._window :]
            self._drain_queued_segments()
            self._vad.accept_waveform(chunk)
            self._drain_queued_segments()
            detected = False
            if hasattr(self._vad, "is_speech_detected"):
                try:
                    detected = bool(self._vad.is_speech_detected())
                except Exception:
                    detected = False
            if detected:
                speech_this = True

        if speech_this:
            was_in_silence = self._silence_samples > 0
            resumed_ms = self._silence_samples / float(self.sample_rate) * 1000.0
            if not self._speech:
                self._speech_samples = 0
            self._speech = True
            self._speech_samples += int(samples.size)
            self._silence_samples = 0
            return VadEvent(
                is_speech=True,
                speech_ended=False,
                silence_ms=0.0,
                speech_resumed=was_in_silence,
                resumed_after_silence_ms=resumed_ms,
            )

        if self._speech:
            self._silence_samples += int(samples.size)
            sil_ms = self._silence_samples / float(self.sample_rate) * 1000.0
            needed = silence_threshold_samples(
                self._speech_samples,
                self.sample_rate,
                WS_FLUSH_SILENCE_MS,
                WS_EARLY_SILENCE_MS,
                WS_COMMITTED_SPEECH_SECONDS,
            )
            if self._silence_samples >= needed:
                self._speech = False
                self._speech_samples = 0
                self._silence_samples = 0
                return VadEvent(is_speech=False, speech_ended=True, silence_ms=sil_ms)
            return VadEvent(is_speech=True, speech_ended=False, silence_ms=sil_ms)

        return VadEvent(is_speech=False, speech_ended=False)

    def reset(self) -> None:
        self._pending = np.zeros(0, dtype=np.float32)
        self._speech = False
        self._speech_samples = 0
        self._silence_samples = 0
        if self._vad is not None and hasattr(self._vad, "reset"):
            try:
                self._vad.reset()
            except Exception:
                pass


def _merge_silence_flags(flags: np.ndarray, hop_s: float, duration: float) -> list[tuple[float, float]]:
    regions: list[tuple[float, float]] = []
    n = int(flags.size)
    if n == 0:
        if duration > 0:
            return [(0.0, float(duration))]
        return []
    i = 0
    while i < n:
        if flags[i]:
            i += 1
            continue
        j = i + 1
        while j < n and not flags[j]:
            j += 1
        start = i * hop_s
        end = min(duration, j * hop_s)
        if end > start:
            regions.append((float(start), float(end)))
        i = j
    if regions and regions[-1][1] < duration - 1e-4 and not flags[-1]:
        regions[-1] = (regions[-1][0], float(duration))
    return regions


def _rms_speech_flags(audio: np.ndarray, sample_rate: int, window: int) -> np.ndarray:
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return np.zeros(0, dtype=bool)
    n = samples.size // window
    if n == 0:
        rms = float(np.sqrt(np.mean(samples * samples)))
        return np.array([rms > 0.01], dtype=bool)
    frames = samples[: n * window].reshape(n, window)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    return rms > 0.01


def _silero_speech_flags(audio: np.ndarray, sample_rate: int, window: int) -> np.ndarray | None:
    model_path = Path(VAD_MODEL_PATH)
    if not model_path.exists():
        return None
    try:
        import sherpa_onnx

        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = str(model_path)
        cfg.silero_vad.threshold = 0.5
        cfg.silero_vad.min_silence_duration = 0.2
        cfg.silero_vad.min_speech_duration = 0.15
        cfg.sample_rate = sample_rate
        vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=10)
        win = int(getattr(cfg.silero_vad, "window_size", None) or window)
    except Exception as e:
        log.warning("Offline VAD init failed: {} — using RMS", e)
        return None

    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    flags: list[bool] = []
    i = 0
    while i + win <= samples.size:
        chunk = samples[i : i + win]
        try:
            vad.accept_waveform(chunk)
        except Exception:
            flags.append(bool(np.sqrt(np.mean(chunk * chunk)) > 0.01))
            i += win
            continue
        detected = False
        if hasattr(vad, "is_speech_detected"):
            try:
                detected = bool(vad.is_speech_detected())
            except Exception:
                detected = False
        if not detected:
            detected = bool(np.sqrt(np.mean(chunk * chunk)) > 0.01)
        flags.append(detected)
        i += win
    return np.array(flags, dtype=bool)


def scan_silences(audio: np.ndarray, sample_rate: int = TARGET_SR) -> list[tuple[float, float]]:
    """Return silence regions (start, end) in seconds for offline chunk planning."""
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    duration = float(samples.size) / float(sample_rate) if samples.size else 0.0
    if duration <= 0:
        return []
    window = 512
    flags = _silero_speech_flags(samples, sample_rate, window)
    hop = window
    if flags is None:
        flags = _rms_speech_flags(samples, sample_rate, window)
        log.info("Offline silence scan used RMS fallback | duration={:.2f}s", duration)
    else:
        hop = window
        log.info("Offline silence scan used Silero | duration={:.2f}s regions pending", duration)
    hop_s = hop / float(sample_rate)
    regions = _merge_silence_flags(flags, hop_s, duration)
    log.info("Offline silence regions={} duration={:.2f}s", len(regions), duration)
    return regions


def choose_hard_cut_silence(
    silences: list[tuple[float, float]],
    target_sec: float,
    search_seconds: float,
    min_silence_ms: int,
) -> tuple[float, float] | None:
    """Choose the nearest sufficiently long silence before a realtime hard-cut target."""
    target = max(0.0, float(target_sec))
    search_start = max(0.0, target - max(0.0, float(search_seconds)))
    min_duration = max(0.0, float(min_silence_ms) / 1000.0)
    candidates: list[tuple[float, float]] = []
    for raw_start, raw_end in silences:
        start, end = float(raw_start), float(raw_end)
        if end <= start or end > target + 1e-6 or end - start + 1e-6 < min_duration:
            continue
        midpoint = (start + end) / 2.0
        if midpoint >= search_start:
            candidates.append((start, end))
    if not candidates:
        return None
    return min(candidates, key=lambda region: (target - (region[0] + region[1]) / 2.0, -(region[1] - region[0])))
