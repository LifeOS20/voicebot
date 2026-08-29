"""Murf Falcon 2 HTTP streaming TTS adapter for Pipecat 1.8.x."""

from __future__ import annotations

import asyncio
import base64
import io
import wave
from collections.abc import AsyncGenerator
from typing import Any

import aiohttp
from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService


class MurfFalcon2TTSService(TTSService):
    """Murf Falcon 2 HTTP TTS service.

    Murf's streaming endpoint may return WAV container bytes even when the
    requested synthesis format is PCM. This adapter normalizes the response
    to raw mono PCM before emitting TTSAudioRawFrame objects downstream.
    """

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str = "Anisha",
        model: str = "falcon-2",
        locale: str = "en-IN",
        base_url: str = "https://in.api.murf.ai",
        sample_rate: int | None = None,
        channel_type: str = "MONO",
        format: str = "PCM",
        request_timeout_seconds: float = 30.0,
        aiohttp_session: aiohttp.ClientSession | None = None,
        settings: TTSSettings | None = None,
        **kwargs: Any,
    ) -> None:
        default_settings = TTSSettings(
            model=model,
            voice=voice_id,
            language=locale,
        )

        if settings is not None:
            default_settings.apply_update(settings)

        # Pipecat establishes the runtime sample rate from StartFrame.
        # Passing None here allows that lifecycle value to remain authoritative.
        super().__init__(
            sample_rate=sample_rate,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._voice_id = voice_id
        self._model = model
        self._locale = locale
        self._base_url = base_url.rstrip("/")
        self._channel_type = channel_type
        self._format = format.upper()
        self._request_timeout_seconds = request_timeout_seconds

        self._session = aiohttp_session
        self._owns_session = aiohttp_session is None

        logger.debug(
            "Murf TTS initialized model={} voice={} locale={} "
            "configured_sample_rate={} format={} channel_type={}",
            self._model,
            self._voice_id,
            self._locale,
            sample_rate,
            self._format,
            self._channel_type,
        )

    async def start(self, frame: StartFrame) -> None:
        """Initialize the service using Pipecat's runtime audio settings."""
        await super().start(frame)

        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

        logger.debug(
            "Murf TTS started runtime_sample_rate={} usable={}",
            self.sample_rate,
            self.is_usable,
        )

    async def cleanup(self) -> None:
        """Release resources owned by this adapter."""
        if self._owns_session and self._session is not None:
            if not self._session.closed:
                await self._session.close()

        await super().cleanup()

    async def set_language(self, locale: str) -> None:
        """Update Murf synthesis locale for subsequent requests."""
        supported = {
            "en-IN",
            "hi-IN",
            "te-IN",
        }

        if locale not in supported:
            raise ValueError(f"Unsupported Murf locale: {locale}")

        self._locale = locale

        # Keep Pipecat's runtime settings in sync.
        if getattr(self, "_settings", None) is not None:
            self._settings.language = locale

        logger.info("Murf TTS language changed to {}", locale)

    @staticmethod
    def _wav_to_pcm(
        audio_bytes: bytes,
    ) -> tuple[bytes, int, int]:
        """Convert a WAV container into raw PCM bytes.

        Returns:
            (pcm_bytes, sample_rate, channel_count)
        """
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav:
            channels = wav.getnchannels()
            sample_rate = wav.getframerate()
            sample_width = wav.getsampwidth()
            pcm = wav.readframes(wav.getnframes())

        if sample_width != 2:
            raise ValueError(
                f"Unsupported Murf WAV sample width: {sample_width * 8}-bit; "
                "expected 16-bit PCM."
            )

        return pcm, sample_rate, channels

    @staticmethod
    def _strip_known_container(audio_bytes: bytes) -> tuple[bytes, int | None, int | None]:
        """Return raw PCM if bytes contain a WAV container.

        For non-WAV bytes, return the original bytes and unknown metadata.
        """
        if audio_bytes[:4] != b"RIFF" or audio_bytes[8:12] != b"WAVE":
            return audio_bytes, None, None

        pcm, sample_rate, channels = MurfFalcon2TTSService._wav_to_pcm(
            audio_bytes
        )
        return pcm, sample_rate, channels

    async def run_tts(
        self,
        text: str,
        context_id: str,
    ) -> AsyncGenerator[Frame | None, None]:
        """Synthesize one utterance and emit playable raw PCM frames."""
        text = (text or "").strip()

        if not text:
            return

        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._owns_session = True

        runtime_sample_rate = self.sample_rate

        logger.info(
            "Murf TTS request context_id={} locale={} sample_rate={} text_length={}",
            context_id,
            self._locale,
            runtime_sample_rate,
            len(text),
        )

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
            "sampleRate": runtime_sample_rate,
        }

        timeout = aiohttp.ClientTimeout(
            total=self._request_timeout_seconds,
            connect=5.0,
            sock_connect=5.0,
            sock_read=self._request_timeout_seconds,
        )

        await self.start_ttfb_metrics()
        await self.start_tts_usage_metrics(text)

        first_audio = True
        response_started = False

        try:
            async with self._session.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout,
            ) as response:

                content_type = response.headers.get(
                    "Content-Type",
                    "",
                ).lower()

                logger.debug(
                    "Murf TTS response status={} content_type={}",
                    response.status,
                    content_type,
                )

                if response.status != 200:
                    body = (await response.text())[:500]

                    logger.error(
                        "Murf TTS failed status={} body={}",
                        response.status,
                        body,
                    )

                    yield ErrorFrame(
                        error=f"Murf TTS HTTP {response.status}"
                    )
                    return

                response_started = True

                yield TTSStartedFrame(context_id=context_id)

                # Murf's endpoint has returned WAV content in our validated
                # direct API test. Read the response as one audio payload,
                # normalize the container if required, then emit PCM.
                audio_bytes = await response.read()

                if not audio_bytes:
                    logger.error("Murf TTS returned an empty audio response.")
                    yield ErrorFrame(
                        error="Murf TTS returned empty audio"
                    )
                    return

                pcm_bytes, returned_sample_rate, returned_channels = (
                    self._strip_known_container(audio_bytes)
                )

                output_sample_rate = (
                    returned_sample_rate or runtime_sample_rate
                )
                output_channels = returned_channels or 1

                if output_channels != 1:
                    logger.error(
                        "Murf TTS returned {} channels; expected mono.",
                        output_channels,
                    )
                    yield ErrorFrame(
                        error=(
                            f"Murf TTS returned {output_channels} channels; "
                            "mono output required"
                        )
                    )
                    return

                if first_audio:
                    await self.stop_ttfb_metrics()
                    first_audio = False

                logger.debug(
                    "Murf TTS audio received bytes={} pcm_bytes={} "
                    "sample_rate={} channels={}",
                    len(audio_bytes),
                    len(pcm_bytes),
                    output_sample_rate,
                    output_channels,
                )

                # Emit reasonably sized PCM chunks downstream.
                chunk_size = 4096

                for offset in range(
                    0,
                    len(pcm_bytes),
                    chunk_size,
                ):
                    chunk = pcm_bytes[
                        offset : offset + chunk_size
                    ]

                    if not chunk:
                        continue

                    yield TTSAudioRawFrame(
                        audio=chunk,
                        sample_rate=output_sample_rate,
                        num_channels=1,
                        context_id=context_id,
                    )

        except asyncio.CancelledError:
            # Normal path during interruption/barge-in or call cancellation.
            raise

        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.error(
                "Murf TTS network/timeout failure: {}",
                exc,
            )

            yield ErrorFrame(
                error="Murf TTS network or timeout failure"
            )

        except (wave.Error, ValueError) as exc:
            logger.error(
                "Murf TTS audio decoding failure: {}",
                exc,
            )

            yield ErrorFrame(
                error="Murf TTS audio decoding failure"
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
            if first_audio:
                await self.stop_ttfb_metrics()

            if response_started:
                yield TTSStoppedFrame(
                    context_id=context_id,
                )