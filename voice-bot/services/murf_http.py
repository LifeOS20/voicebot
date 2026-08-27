"""Murf Falcon 2 HTTP streaming TTS adapter."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

import aiohttp
from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService


class MurfFalcon2TTSService(TTSService):
    """Murf Falcon 2 REST streaming service.

    The service emits PCM chunks as they arrive, so Pipecat can begin
    playback without waiting for the complete utterance.
    """

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str = "Anisha",
        model: str = "falcon-2",
        locale: str = "en-IN",
        base_url: str = "https://in.api.murf.ai",
        sample_rate: int = 8000,
        channel_type: str = "MONO",
        format: str = "PCM",
        request_timeout_seconds: float = 30.0,
        aiohttp_session: aiohttp.ClientSession | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            sample_rate=sample_rate,
            **kwargs,
        )

        self._api_key = api_key
        self._voice_id = voice_id
        self._model = model
        self._locale = locale
        self._base_url = base_url.rstrip("/")
        self._channel_type = channel_type
        self._format = format
        self._request_timeout = aiohttp.ClientTimeout(
            total=request_timeout_seconds,
            connect=5.0,
            sock_connect=5.0,
            sock_read=request_timeout_seconds,
        )

        self._session = aiohttp_session
        self._owns_session = aiohttp_session is None

        self.set_model_name(model)
        self.set_voice(voice_id)

    async def start(self, frame: StartFrame):
        await super().start(frame)

        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

    async def cleanup(self):
        await super().cleanup()

        if self._owns_session and self._session:
            if not self._session.closed:
                await self._session.close()

    async def set_language(
        self,
        locale: str,
    ) -> None:
        supported = {
            "en-IN",
            "hi-IN",
            "te-IN",
        }

        if locale not in supported:
            raise ValueError(
                f"Unsupported Murf locale: {locale}"
            )

        self._locale = locale

    async def run_tts(
        self,
        text: str,
        context_id: str,
    ) -> AsyncGenerator[Frame | None, None]:
        text = (text or "").strip()

        if not text:
            return

        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._owns_session = True

        url = f"{self._base_url}/v1/speech/stream"

        headers = {
            "api-key": self._api_key,
            "Accept": "application/octet-stream",
            "Content-Type": "application/json",
        }

        payload = {
            "text": text,
            "voiceId": self._voice_id,
            "model": self._model,
            "locale": self._locale,
            "channelType": self._channel_type,
            "format": self._format,
            "sampleRate": self.sample_rate,
        }

        await self.start_ttfb_metrics()
        await self.start_tts_usage_metrics(text)

        first_chunk = True

        try:
            async with self._session.post(
                url,
                headers=headers,
                json=payload,
                timeout=self._request_timeout,
            ) as response:

                if response.status != 200:
                    body = (await response.text())[:500]
                    logger.error(
                        "Murf TTS failed: status={} body={}",
                        response.status,
                        body,
                    )
                    yield ErrorFrame(
                        error=f"Murf TTS HTTP {response.status}"
                    )
                    return

                async for chunk in response.content.iter_chunked(4096):
                    if not chunk:
                        continue

                    if first_chunk:
                        await self.stop_ttfb_metrics()
                        first_chunk = False

                    yield TTSAudioRawFrame(
                        audio=chunk,
                        sample_rate=self.sample_rate,
                        num_channels=1,
                        context_id=context_id,
                    )

        except asyncio.CancelledError:
            # Expected during caller barge-in/call cancellation.
            raise

        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.error(
                "Murf TTS network/timeout failure: {}",
                exc,
            )
            yield ErrorFrame(
                error="Murf TTS network or timeout failure"
            )

        except Exception as exc:
            logger.exception(
                "Unexpected Murf TTS failure: {}",
                exc,
            )
            yield ErrorFrame(
                error="Murf TTS failure"
            )

        finally:
            if first_chunk:
                await self.stop_ttfb_metrics()

            yield TTSStoppedFrame(
                context_id=context_id,
            )
