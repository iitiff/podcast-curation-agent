"""Transcription provider implementations."""
from __future__ import annotations

import logging

import httpx

from .base import BaseLLMProvider, BaseTranscriptionProvider, TranscriptResult

log = logging.getLogger(__name__)


class NullTranscriptionProvider(BaseTranscriptionProvider):
    """Returns empty transcript — used when no transcription API is configured."""

    async def transcribe(self, episode_url: str, description: str = "") -> TranscriptResult:
        return TranscriptResult(text="", confidence="none", source="none")


class WhisperTranscriptionProvider(BaseTranscriptionProvider):
    """OpenAI Whisper transcription provider."""

    def __init__(self, api_key: str, max_audio_mb: float = 25.0) -> None:
        self.api_key = api_key
        self.max_audio_bytes = int(max_audio_mb * 1024 * 1024)

    async def transcribe(self, episode_url: str, description: str = "") -> TranscriptResult:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                head = await client.head(episode_url)
                content_length = int(head.headers.get("content-length", 0))
                if content_length > self.max_audio_bytes:
                    log.info("Audio too large (%d bytes), skipping Whisper", content_length)
                    return TranscriptResult(text="", confidence="none", source="none")
                resp = await client.get(episode_url)
                resp.raise_for_status()
                audio_data = resp.content
        except Exception as exc:
            log.warning("Failed to fetch audio from %s: %s", episode_url, exc)
            return TranscriptResult(text="", confidence="none", source="none")

        import openai
        client = openai.AsyncOpenAI(api_key=self.api_key)
        try:
            result = await client.audio.transcriptions.create(
                model="whisper-1",
                file=("audio.mp3", audio_data, "audio/mpeg"),
            )
            return TranscriptResult(
                text=result.text,
                confidence="high",
                source="whisper",
            )
        except Exception as exc:
            log.warning("Whisper transcription failed: %s", exc)
            return TranscriptResult(text="", confidence="none", source="none")


class DescriptionFallbackProvider(BaseTranscriptionProvider):
    """Uses episode description as a transcript substitute."""

    async def transcribe(self, episode_url: str, description: str = "") -> TranscriptResult:
        if not description:
            return TranscriptResult(text="", confidence="none", source="none")
        return TranscriptResult(
            text=description[:3000],
            confidence="low",
            source="description",
        )


class LLMDescriptionEnhancer:
    """Uses an LLM to expand a short description into a richer pseudo-transcript."""

    def __init__(self, llm: BaseLLMProvider) -> None:
        self.llm = llm

    async def enhance(self, show: str, episode: str, description: str) -> TranscriptResult:
        from .base import LLMMessage
        prompt = (
            f"You are a podcast analyst. Given the show '{show}', episode '{episode}', "
            f"and this description:\n\n{description[:1000]}\n\n"
            "Write a 200-word expanded summary of what this episode likely covers, "
            "including probable key ideas and any named guests or companies."
        )
        try:
            resp = await self.llm.complete(
                messages=[LLMMessage(role="user", content=prompt)],
                max_tokens=400,
            )
            return TranscriptResult(
                text=resp.content,
                confidence="low",
                source="description",
            )
        except Exception as exc:
            log.warning("LLM description enhancement failed: %s", exc)
            return TranscriptResult(text=description[:1000], confidence="low", source="description")


class CascadeTranscriptionProvider(BaseTranscriptionProvider):
    """Tries Whisper → description fallback → empty."""

    def __init__(
        self,
        openai_api_key: str | None = None,
        llm: BaseLLMProvider | None = None,
        enable_whisper: bool = False,
    ) -> None:
        self._whisper = WhisperTranscriptionProvider(openai_api_key) if (openai_api_key and enable_whisper) else None
        self._llm_enhancer = LLMDescriptionEnhancer(llm) if llm else None
        self._description = DescriptionFallbackProvider()

    async def transcribe(self, episode_url: str, description: str = "") -> TranscriptResult:
        if self._whisper:
            result = await self._whisper.transcribe(episode_url)
            if result.text:
                return result
        if self._llm_enhancer and description:
            return await self._llm_enhancer.enhance("", "", description)
        return await self._description.transcribe(episode_url, description)
