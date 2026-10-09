"""Utterance-scoped encoder cache. Not a whole-session cache."""
from __future__ import annotations


def full_window_count(n_samples: int, window_samples: int) -> int:
    """How many complete encoder windows fit in ``n_samples``."""
    if int(window_samples) <= 0 or int(n_samples) <= 0:
        return 0
    return int(n_samples) // int(window_samples)


class UtteranceEncoderCache:
    """cache[session][utt] keyed by audio_start_sample + fingerprint."""

    def __init__(self, fingerprint: str = ""):
        self.fingerprint = str(fingerprint or "")
        self._sessions: dict[str, dict[int, dict]] = {}

    def _bucket(self, session_id: str | None, utt_id: int | None):
        if not session_id or utt_id is None:
            return None
        sess = self._sessions.get(session_id)
        if not sess:
            return None
        return sess.get(int(utt_id))

    def peek_chunks(
        self,
        session_id: str | None,
        utt_id: int | None,
        audio_start_sample: int,
        fingerprint: str | None = None,
    ) -> dict | None:
        """Return chunks only when the origin matches. Never replaces the bucket."""
        bucket = self._bucket(session_id, utt_id)
        if bucket is None:
            return None
        fp = str(fingerprint if fingerprint is not None else self.fingerprint)
        if int(bucket.get("audio_start_sample", -1)) != int(audio_start_sample):
            return None
        if str(bucket.get("fingerprint", "")) != fp:
            return None
        return bucket["chunks"]

    def ensure_origin(
        self,
        session_id: str | None,
        utt_id: int | None,
        audio_start_sample: int,
        fingerprint: str | None = None,
    ) -> dict | None:
        """Get or create the origin bucket. A mismatched start does not wipe it."""
        if not session_id or utt_id is None:
            return None
        fp = str(fingerprint if fingerprint is not None else self.fingerprint)
        start = int(audio_start_sample)
        sess = self._sessions.setdefault(session_id, {})
        bucket = sess.get(int(utt_id))
        if bucket is None:
            bucket = {
                "audio_start_sample": start,
                "fingerprint": fp,
                "chunks": {},
                "raw_windows": [],
                "revision": 0,
            }
            sess[int(utt_id)] = bucket
            return bucket["chunks"]
        if int(bucket.get("audio_start_sample", -1)) != start or str(bucket.get("fingerprint", "")) != fp:
            return None
        return bucket["chunks"]

    def window(
        self,
        session_id: str | None,
        utt_id: int | None,
        audio_start_sample: int,
        fingerprint: str | None = None,
    ) -> dict | None:
        if not session_id or utt_id is None:
            return None
        fp = str(fingerprint if fingerprint is not None else self.fingerprint)
        start = int(audio_start_sample)
        sess = self._sessions.setdefault(session_id, {})
        bucket = sess.get(int(utt_id))
        if (
            bucket is None
            or int(bucket.get("audio_start_sample", -1)) != start
            or str(bucket.get("fingerprint", "")) != fp
        ):
            bucket = {
                "audio_start_sample": start,
                "fingerprint": fp,
                "chunks": {},
                "raw_windows": [],
                "revision": 0,
            }
            sess[int(utt_id)] = bucket
        return bucket["chunks"]

    def meta(self, session_id: str | None, utt_id: int | None) -> dict | None:
        return self._bucket(session_id, utt_id)

    def bump_revision(self, session_id: str | None, utt_id: int | None) -> int:
        bucket = self._bucket(session_id, utt_id)
        if bucket is None:
            return 0
        bucket["revision"] = int(bucket.get("revision") or 0) + 1
        return int(bucket["revision"])

    def set_resume(self, session_id: str | None, utt_id: int | None, resume: dict | None) -> None:
        bucket = self._bucket(session_id, utt_id)
        if bucket is None:
            return
        if resume is None:
            bucket.pop("resume", None)
            return
        bucket["resume"] = resume

    def get_resume(self, session_id: str | None, utt_id: int | None) -> dict | None:
        bucket = self._bucket(session_id, utt_id)
        if bucket is None:
            return None
        resume = bucket.get("resume")
        return resume if isinstance(resume, dict) else None

    def drop_utt(self, session_id: str | None, utt_id: int | None) -> None:
        if not session_id or utt_id is None:
            return
        sess = self._sessions.get(session_id)
        if not sess:
            return
        sess.pop(int(utt_id), None)
        if not sess:
            self._sessions.pop(session_id, None)

    def drop_session(self, session_id: str | None) -> None:
        if not session_id:
            return
        self._sessions.pop(session_id, None)
