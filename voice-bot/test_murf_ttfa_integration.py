"""End-to-end proof against the REAL `MurfFalcon2TTSService.run_tts()`
method (not just the parser in isolation): first audio must arrive before
the simulated network response finishes sending.

No live Murf API is reachable from this environment, so the HTTP layer is
mocked -- but everything above that line (the actual shipped `run_tts`
async generator, its chunking loop, its metrics timing calls) is the real
code path that will run in production. The mock simulates a slow network
by sleeping between chunks, exactly the shape of a real streaming response.

Run: python3 test_murf_ttfa_integration.py
"""

from __future__ import annotations

import asyncio
import io
import struct
import time
import unittest
import wave

from pipecat.frames.frames import TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame

from services.murf_http import MurfFalcon2TTSService


def _build_wav(pcm: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buf.getvalue()


class _FakeContent:
    """Mimics aiohttp's StreamReader.iter_chunked(), sleeping between
    chunks to simulate a real network response arriving progressively."""

    def __init__(self, chunks: list[bytes], delay_s: float):
        self._chunks = chunks
        self._delay_s = delay_s

    async def iter_chunked(self, n: int):
        for chunk in self._chunks:
            await asyncio.sleep(self._delay_s)
            yield chunk


class _FakeResponse:
    def __init__(self, chunks: list[bytes], delay_s: float):
        self.status = 200
        self.headers = {"Content-Type": "application/octet-stream"}
        self.content = _FakeContent(chunks, delay_s)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    def __init__(self, response: _FakeResponse):
        self._response = response
        self.closed = False

    def post(self, *args, **kwargs):
        return self._response


class TestMurfTimeToFirstAudio(unittest.TestCase):
    def test_first_audio_frame_arrives_before_stream_finishes(self):
        sample_rate = 24000
        # ~2 seconds of audio, split into 6 simulated network chunks with a
        # deliberate delay between each -- if run_tts buffered the whole
        # response before yielding anything (the pre-fix behavior), the
        # first TTSAudioRawFrame could only appear after all 6 delays have
        # elapsed. If it streams correctly, it must appear after roughly
        # one chunk's worth of delay.
        n_samples = sample_rate * 2
        pcm = struct.pack(
            f"<{n_samples}h", *[(i * 31) % 20000 - 10000 for i in range(n_samples)]
        )
        wav_bytes = _build_wav(pcm, sample_rate)

        n_network_chunks = 6
        per_chunk_delay = 0.05  # 50ms between simulated network chunks
        step = max(1, len(wav_bytes) // n_network_chunks)
        network_chunks = [
            wav_bytes[i : i + step] for i in range(0, len(wav_bytes), step)
        ]

        fake_response = _FakeResponse(network_chunks, per_chunk_delay)
        fake_session = _FakeSession(fake_response)

        tts = MurfFalcon2TTSService(
            api_key="fake",
            sample_rate=sample_rate,
            aiohttp_session=fake_session,
        )

        frame_timestamps: list[tuple[float, type]] = []
        reconstructed_pcm = bytearray()

        async def run():
            start = time.monotonic()
            async for frame in tts.run_tts("hello, this is a latency test", "ctx-1"):
                now = time.monotonic() - start
                frame_timestamps.append((now, type(frame)))
                if isinstance(frame, TTSAudioRawFrame):
                    reconstructed_pcm.extend(frame.audio)

        asyncio.run(run())

        audio_frame_times = [
            t for t, cls in frame_timestamps if cls is TTSAudioRawFrame
        ]
        total_stream_time = n_network_chunks * per_chunk_delay

        self.assertGreater(
            len(audio_frame_times), 1, "expected multiple incremental audio frames"
        )

        first_audio_time = audio_frame_times[0]
        last_audio_time = audio_frame_times[-1]

        print(
            f"\n  total simulated network time: {total_stream_time*1000:.0f}ms\n"
            f"  first TTSAudioRawFrame at:    {first_audio_time*1000:.0f}ms\n"
            f"  last TTSAudioRawFrame at:     {last_audio_time*1000:.0f}ms\n"
            f"  audio frames emitted:         {len(audio_frame_times)}"
        )

        # The real claim: first audio must show up well before the full
        # response has finished arriving -- not clustered at the very end.
        self.assertLess(
            first_audio_time,
            total_stream_time * 0.6,
            "first audio frame arrived too late -- looks like it's still "
            "buffering the whole response before yielding anything",
        )

        # No audio gaps/regressions: the reconstructed PCM must be
        # byte-identical to the original, in order, nothing dropped or
        # duplicated.
        self.assertEqual(bytes(reconstructed_pcm), pcm)

        # Started/stopped frames still present exactly once each.
        self.assertEqual(
            sum(1 for _, cls in frame_timestamps if cls is TTSStartedFrame), 1
        )
        self.assertEqual(
            sum(1 for _, cls in frame_timestamps if cls is TTSStoppedFrame), 1
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
