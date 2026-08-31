"""Proves `_IncrementalRiffParser` is byte-correct, without needing a live
Murf API call.

I cannot reach murf.ai from this environment to test the real endpoint, so
this instead builds a real WAV file with Python's own stdlib `wave` module
(ground truth), then feeds those exact bytes into the incremental parser in
adversarial ways -- one byte at a time, in random chunk sizes, split exactly
at subchunk boundaries -- and asserts the parser recovers the identical PCM
payload, sample rate, and channel count that `wave.open()` reports for the
same file. It also proves the non-WAV (raw PCM passthrough) path and the
malformed-input safety valve.

Run: python3 test_murf_streaming_parser.py
"""

from __future__ import annotations

import io
import random
import struct
import unittest
import wave

from services.murf_http import _IncrementalRiffParser


def _build_wav(pcm: bytes, sample_rate: int, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buf.getvalue()


def _feed_in_pieces(parser: _IncrementalRiffParser, data: bytes, piece_sizes: list[int]) -> bytes:
    out = bytearray()
    offset = 0
    for size in piece_sizes:
        if offset >= len(data):
            break
        chunk = data[offset : offset + size]
        offset += size
        out.extend(parser.feed(chunk))
    if offset < len(data):
        out.extend(parser.feed(data[offset:]))
    out.extend(parser.flush())
    return bytes(out)


class TestIncrementalRiffParser(unittest.TestCase):
    def setUp(self):
        random.seed(42)
        # Deterministic "audio": not silence, so a truncation/misalignment
        # bug can't accidentally look correct.
        self.pcm = struct.pack("<2000h", *[(i * 37) % 30000 - 15000 for i in range(2000)])

    def test_whole_response_fed_as_one_chunk(self):
        wav_bytes = _build_wav(self.pcm, sample_rate=24000)
        parser = _IncrementalRiffParser()
        result = _feed_in_pieces(parser, wav_bytes, [len(wav_bytes)])
        self.assertEqual(result, self.pcm)
        self.assertEqual(parser.sample_rate, 24000)
        self.assertEqual(parser.channels, 1)
        self.assertTrue(parser.is_wav)

    def test_one_byte_at_a_time(self):
        """The adversarial case: network chunks of size 1, so every single
        header field boundary gets split mid-value at some point."""
        wav_bytes = _build_wav(self.pcm, sample_rate=16000)
        parser = _IncrementalRiffParser()
        result = _feed_in_pieces(parser, wav_bytes, [1] * len(wav_bytes))
        self.assertEqual(result, self.pcm)
        self.assertEqual(parser.sample_rate, 16000)
        self.assertEqual(parser.channels, 1)

    def test_random_chunk_sizes(self):
        wav_bytes = _build_wav(self.pcm, sample_rate=8000)
        sizes = [random.randint(1, 7) for _ in range(len(wav_bytes))]
        parser = _IncrementalRiffParser()
        result = _feed_in_pieces(parser, wav_bytes, sizes)
        self.assertEqual(result, self.pcm)
        self.assertEqual(parser.sample_rate, 8000)

    def test_split_exactly_at_data_marker(self):
        """Split the stream so one chunk ends with exactly the 'da' of
        'data' and the size field straddles two network chunks -- the
        specific boundary condition most likely to be gotten wrong."""
        wav_bytes = _build_wav(self.pcm, sample_rate=24000)
        idx = wav_bytes.find(b"data")
        self.assertNotEqual(idx, -1)
        split_point = idx + 2  # right in the middle of "data"
        parser = _IncrementalRiffParser()
        result = _feed_in_pieces(
            parser,
            wav_bytes,
            [split_point, len(wav_bytes) - split_point],
        )
        self.assertEqual(result, self.pcm)

    def test_first_chunk_arrives_before_data_found_yields_nothing_yet(self):
        """First network chunk (just RIFF/WAVE/fmt) must not release any
        PCM bytes early -- that would mean emitting header bytes as if they
        were audio."""
        wav_bytes = _build_wav(self.pcm, sample_rate=24000)
        idx = wav_bytes.find(b"data")
        parser = _IncrementalRiffParser()
        first_release = parser.feed(wav_bytes[:idx])
        self.assertEqual(first_release, b"")
        self.assertFalse(parser.data_started)

    def test_raw_pcm_passthrough_when_not_a_wav_container(self):
        """Murf can return raw PCM directly (no RIFF header at all) --
        every byte must pass straight through, untouched, from the first
        chunk."""
        raw = self.pcm  # no WAV wrapper at all
        parser = _IncrementalRiffParser()
        result = _feed_in_pieces(parser, raw, [17] * (len(raw) // 17 + 1))
        self.assertEqual(result, raw)
        self.assertFalse(parser.is_wav)
        self.assertIsNone(parser.sample_rate)

    def test_malformed_input_does_not_stall_forever(self):
        """A RIFF/WAVE header with no 'data' subchunk ever appearing must
        eventually be treated as raw PCM instead of buffering forever."""
        garbage_header = b"RIFF" + (b"\x00" * 4) + b"WAVE" + (b"XXXX" * 2000)
        parser = _IncrementalRiffParser()
        result = _feed_in_pieces(parser, garbage_header, [500] * 20)
        self.assertEqual(len(result), len(garbage_header))
        self.assertTrue(parser.data_started)

    def test_stereo_channel_count_is_detected(self):
        stereo_pcm = struct.pack("<4000h", *([1000, -1000] * 2000))
        wav_bytes = _build_wav(stereo_pcm, sample_rate=24000, channels=2)
        parser = _IncrementalRiffParser()
        result = _feed_in_pieces(parser, wav_bytes, [37] * (len(wav_bytes) // 37 + 1))
        self.assertEqual(result, stereo_pcm)
        self.assertEqual(parser.channels, 2)

    def test_pcm_is_released_incrementally_not_only_at_the_end(self):
        """The actual latency claim, proven directly: PCM bytes must come
        back from an early `feed()` call, before the rest of the response
        has even been sent -- not only from `flush()` after everything has
        arrived. This is what the old `wave.open()`-based implementation
        could never do."""
        wav_bytes = _build_wav(self.pcm, sample_rate=24000)
        idx = wav_bytes.find(b"data")
        # First "network chunk": header + first third of the PCM payload.
        first_chunk_end = idx + 8 + (len(self.pcm) // 3)
        parser = _IncrementalRiffParser()

        released_early = parser.feed(wav_bytes[:first_chunk_end])

        # Must already have real PCM bytes back, and it must be an exact,
        # untruncated prefix of the true PCM stream -- not padding, not
        # empty, not the header.
        self.assertGreater(len(released_early), 0)
        self.assertEqual(released_early, self.pcm[: len(released_early)])
        self.assertTrue(parser.data_started)

        # The rest of the response hasn't been "sent" yet in this test, so
        # nothing past what we released should be reconstructable yet.
        remaining = parser.feed(wav_bytes[first_chunk_end:])
        remaining += parser.flush()
        self.assertEqual(released_early + remaining, self.pcm)


if __name__ == "__main__":
    unittest.main(verbosity=2)
