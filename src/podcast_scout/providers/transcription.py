"""Transcription providers."""
from __future__ import annotations

from .base import BaseTranscriptionProvider


class NullTranscriptionProvider(BaseTranscriptionProvider):
    """Used when transcription is disabled or no key is available."""
    async def transcribe(self, audio_url: str, max_minutes: float = 10.0) -> str:
        return ""
