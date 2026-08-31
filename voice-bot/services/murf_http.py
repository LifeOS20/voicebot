"""Murf Falcon 2 HTTP streaming TTS adapter for Pipecat 1.8.x."""

from __future__ import annotations

import asyncio
import struct
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


class _IncrementalRiffParser:
    """Strips a WAV/RIFF container from a byte stream as it arrives, without
    ever needing the complete file in memory.

    FIXED (critical latency bug): the previous implementation called
    `wave.open(io.BytesIO(audio_bytes), "rb")` on the response body. The
    stdlib `wave` module requires a complete, seekable buffer to do that --
    it reads the RIFF size field and seeks to pull out frame counts -- so
    the calling code had no choice but to buffer Murf's *entire* response
    with `await response.read()` before it could parse anything or emit a
    single frame of audio. That's why time-to-first-audio for Murf was
    effectively "however long the whole utterance takes to synthesize and
    download," not "however long the first chunk takes."

    This class does the one thing `wave` can't: walk RIFF subchunks
    incrementally as bytes arrive over the wire, so raw PCM payload bytes
    can be handed off the moment they're available, chunk to chunk. It
    still recovers the real sample rate and channel count from the `fmt `
    subchunk when one is present -- Murf's docs note it can return WAV
    container bytes even when PCM was requested, so that check is preserved
    exactly, not relaxed.

    Usage: call `feed(chunk)` for each network chunk as it arrives; it
    returns whatever newly-available raw PCM bytes that chunk unlocked
    (empty bytes while still buffering header). After the stream ends, call
    `flush()` for any tail bytes still held internally (there normally
    aren't any once `data_started` is True, since `feed` hands off
    everything immediately at that point -- `flush` exists purely so a
    malformed/truncated response can't silently lose its last few bytes).
    """

    # Never scan more than this many header bytes before giving up on
    # detecting a container and just treating everything as raw PCM. A real
    # WAV header for this use case (mono 16-bit PCM, a handful of standard
    # subchunks) is well under 100 bytes; this is a generous safety margin,
    # not a tuned value -- it exists so a malformed response can't stall
    # first-audio forever waiting for a `data` marker that never comes.
    MAX_HEADER_SCAN_BYTES = 4096

    def __init__(self) -> None:
        self._header_buf = bytearray()
        self._offset = 0
        self.data_started = False
        self.is_wav: bool | None = None  # None = not yet determined
        self.sample_rate: int | None = None
        self.channels: int | None = None

    def feed(self, chunk: bytes) -> bytes:
        if self.data_started:
            return chunk

        self._header_buf.extend(chunk)

        if self.is_wav is None:
            if len(self._header_buf) < 12:
                return b""  # not enough yet to even check the magic bytes
            self.is_wav = (
                self._header_buf[:4] == b"RIFF"
                and self._header_buf[8:12] == b"WAVE"
            )
            if not self.is_wav:
                # Not a RIFF container at all -- Murf returned raw PCM
                # directly. Everything buffered so far, plus everything
                # from here on, is audio payload.
                pending = bytes(self._header_buf)
                self._header_buf = bytearray()
                self.data_started = True
                return pending
            self._offset = 12

        # Walk subchunks: 4-byte ID + 4-byte little-endian size + payload
        # (word-padded). Stop as soon as `data` is found, or as soon as we
        # run out of buffered bytes to safely parse the next subchunk header.
        while True:
            if len(self._header_buf) - self._offset < 8:
                break  # need more bytes for the next subchunk header

            subchunk_id = bytes(self._header_buf[self._offset : self._offset + 4])
            (subchunk_size,) = struct.unpack_from(
                "<I", self._header_buf, self._offset + 4
            )
            payload_start = self._offset + 8

            if subchunk_id == b"data":
                # Found it. Everything from here to the end of what we've
                # buffered is already audio payload; hand it all off now.
                pending = bytes(self._header_buf[payload_start:])
                self._header_buf = bytearray()
                self.data_started = True
                return pending

            if len(self._header_buf) - payload_start < subchunk_size:
                break  # this subchunk's payload hasn't fully arrived yet

            if subchunk_id == b"fmt " and subchunk_size >= 16:
                # AudioFormat(2) NumChannels(2) SampleRate(4) ...
                self.channels, self.sample_rate = struct.unpack_from(
                    "<xxHI", self._header_buf, payload_start
                )

            self._offset = payload_start + subchunk_size + (subchunk_size & 1)

        if len(self._header_buf) > self.MAX_HEADER_SCAN_BYTES:
            logger.warning(
                "Murf TTS: no RIFF 'data' subchunk found in first {} bytes; "
                "treating buffered response as raw PCM instead of stalling.",
                len(self._header_buf),
            )
            pending = bytes(self._header_buf)
            self._header_buf = bytearray()
            self.data_started = True
            return pending

        return b""

    def flush(self) -> bytes:
        if self._header_buf:
            pending = bytes(self._header_buf)
            self._header_buf = bytearray()
            return pending
        return b""


class MurfFalcon2TTSService(TTSService):
    """Murf Falcon 2 HTTP TTS service.

    Murf's streaming endpoint may return WAV container bytes even when the
    requested synthesis format is PCM. This adapter normalizes the response
    to raw mono PCM before emitting TTSAudioRawFrame objects downstream --
    incrementally, as bytes arrive, never buffering the full response first.
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

    async def run_tts(
        self,
        text: str,
        context_id: str,
    ) -> AsyncGenerator[Frame | None, None]:
        """Synthesize one utterance and emit playable raw PCM frames.

        Streams audio downstream as it arrives from Murf, chunk by chunk --
        it never waits for the full response before yielding the first
        frame. See `_IncrementalRiffParser` for why that was previously
        impossible with a WAV-container response.
        """
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

                # Stream the response as it arrives instead of buffering it.
                # `_IncrementalRiffParser` strips a WAV container (if Murf
                # sent one) chunk by chunk, so the first playable PCM bytes
                # can be emitted as soon as they're available -- not after
                # the entire utterance has finished downloading.
                parser = _IncrementalRiffParser()
                pending = bytearray()
                chunk_size = 4096
                total_network_bytes = 0
                total_pcm_bytes = 0

                async for network_chunk in response.content.iter_chunked(8192):
                    if not network_chunk:
                        continue

                    total_network_bytes += len(network_chunk)
                    pending.extend(parser.feed(network_chunk))

                    while len(pending) >= chunk_size:
                        piece = bytes(pending[:chunk_size])
                        del pending[:chunk_size]
                        total_pcm_bytes += len(piece)

                        if first_audio:
                            await self.stop_ttfb_metrics()
                            first_audio = False

                        yield TTSAudioRawFrame(
                            audio=piece,
                            sample_rate=(parser.sample_rate or runtime_sample_rate),
                            num_channels=1,
                            context_id=context_id,
                        )

                # Flush whatever's left: a final short chunk from the loop
                # above, plus anything the parser was still holding if the
                # stream ended before a `data` subchunk was ever found
                # (malformed/empty response).
                pending.extend(parser.flush())

                if pending:
                    total_pcm_bytes += len(pending)

                    if first_audio:
                        await self.stop_ttfb_metrics()
                        first_audio = False

                    yield TTSAudioRawFrame(
                        audio=bytes(pending),
                        sample_rate=(parser.sample_rate or runtime_sample_rate),
                        num_channels=1,
                        context_id=context_id,
                    )

                if total_pcm_bytes == 0:
                    logger.error("Murf TTS returned an empty audio response.")
                    yield ErrorFrame(
                        error="Murf TTS returned empty audio"
                    )
                    return

                if parser.channels is not None and parser.channels != 1:
                    logger.error(
                        "Murf TTS returned {} channels; expected mono.",
                        parser.channels,
                    )
                    yield ErrorFrame(
                        error=(
                            f"Murf TTS returned {parser.channels} channels; "
                            "mono output required"
                        )
                    )
                    return

                logger.debug(
                    "Murf TTS audio streamed network_bytes={} pcm_bytes={} "
                    "sample_rate={} is_wav={}",
                    total_network_bytes,
                    total_pcm_bytes,
                    parser.sample_rate or runtime_sample_rate,
                    parser.is_wav,
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

        except (struct.error, ValueError) as exc:
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