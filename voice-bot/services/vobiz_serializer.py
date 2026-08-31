"""
Vobiz Frame Serializer with simple, robust interruption handling.

Key features:
- Post-interruption audio drop window to eliminate stale TTS stragglers
- Proper µ-law <-> PCM conversion with resampling
- DTMF support
"""

import base64
import json
import audioop
import time
from loguru import logger
from pipecat.audio.utils import ulaw_to_pcm, create_stream_resampler
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    InputAudioRawFrame,
    StartFrame,
    InterruptionFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    InputDTMFFrame,
)
from pipecat.audio.dtmf.types import KeypadEntry


# How long after sending clearAudio to keep dropping outgoing audio frames.
# This addresses the Pipecat frame ordering issue where AudioRawFrames from a
# cancelled utterance can still be in-flight when InterruptionFrame arrives.
# Increased to 1.0s to account for network RTT + pacing + cancellation latency.
_POST_CLEAR_DROP_WINDOW_S = 1.0


class VobizFrameSerializer(FrameSerializer):
    def __init__(self, stream_id: str, sample_rate: int = 8000):
        self._stream_id = stream_id
        self._vobiz_sample_rate = sample_rate
        self._sample_rate = 0

        self._input_resampler = create_stream_resampler()
        self._output_resampler = create_stream_resampler()

        # Simple interruption handling: drop window after clearAudio
        self._drop_audio_until = 0.0

    async def setup(self, frame: StartFrame):
        self._sample_rate = frame.audio_in_sample_rate
        logger.debug(
            "[VobizSerializer] Setup complete stream_id={} pipeline_sample_rate={} vobiz_sample_rate={}",
            self._stream_id,
            self._sample_rate,
            self._vobiz_sample_rate,
        )

    def on_interruption(self) -> None:
        """Call when barge-in detected - activates drop window."""
        self._drop_audio_until = time.monotonic() + _POST_CLEAR_DROP_WINDOW_S
        logger.debug(
            "[VobizSerializer] Interruption: drop window activated for {}s stream_id={}",
            _POST_CLEAR_DROP_WINDOW_S,
            self._stream_id,
        )

    def next_generation(self) -> int:
        """Call when a new user turn starts - clears any pending drop window."""
        self._drop_audio_until = 0.0
        logger.debug(
            "[VobizSerializer] New generation started, drop window cleared stream_id={}",
            self._stream_id,
        )
        return 1

    async def serialize(self, frame: Frame) -> str | bytes | None:
        # InterruptionFrame: send clearAudio and activate drop window
        if isinstance(frame, InterruptionFrame):
            self.on_interruption()
            return json.dumps({"event": "clearAudio", "streamId": self._stream_id})

        # AudioRawFrame: apply drop window
        if isinstance(frame, AudioRawFrame):
            # Drop if in post-interruption guard window
            if time.monotonic() < self._drop_audio_until:
                logger.warning(
                    "[VobizSerializer] Dropped {} bytes in post-interruption guard window stream_id={}",
                    len(frame.audio),
                    self._stream_id,
                )
                return None

            # Resample to 8000Hz if needed
            if frame.sample_rate != self._vobiz_sample_rate:
                audio_data = await self._output_resampler.resample(
                    frame.audio,
                    frame.sample_rate,
                    self._vobiz_sample_rate,
                )
            else:
                audio_data = frame.audio

            if not audio_data:
                logger.warning("[VobizSerializer] Serialized audio data is empty")
                return None

            # Convert PCM (16-bit) to Mu-Law (8-bit)
            try:
                ulaw_data = audioop.lin2ulaw(audio_data, 2)
            except Exception as e:
                logger.error("[VobizSerializer] Error encoding audio to mu-law: {}", e)
                return None

            payload = base64.b64encode(ulaw_data).decode("utf-8")
            return json.dumps(
                {
                    "event": "playAudio",
                    "streamId": self._stream_id,
                    "media": {
                        "contentType": "audio/x-mulaw",
                        "sampleRate": 8000,
                        "payload": payload,
                    },
                }
            )

        # Custom events (e.g., call transfer)
        if isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            return json.dumps(frame.message)
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        # Convert a Vobiz WebSocket message into formats the Pipecat pipeline can understand.
        try:
            message = json.loads(data)
        except json.JSONDecodeError:
            logger.warning("Failed to parse WebSocket JSON payload")
            return None

        # Incoming audio from the caller, Vobiz sends that voice data to server
        if message.get("event") == "media":
            media = message.get("media", {})
            payload_base64 = media.get("payload")
            if not payload_base64:
                return None

            # The audio arrives as encoded text. We convert that text back into real audio bytes.
            payload = base64.b64decode(payload_base64)

            # We check what type of audio Vobiz sent, either µ-law (phone format) or raw PCM
            content_type = str(media.get("contentType", "audio/x-mulaw")).lower()
            input_rate = int(media.get("sampleRate", self._vobiz_sample_rate))

            if "audio/x-l16" in content_type:
                # If the audio is in PCM format, we only adjust the sample rate if it does not match the pipeline
                if input_rate == self._sample_rate:
                    deserialized_data = payload
                else:
                    deserialized_data = await self._input_resampler.resample(payload, input_rate, self._sample_rate)
            else:
                # If the audio is in phone format, we convert it into normal PCM and adjust the sample rate so the AI can use it
                deserialized_data = await ulaw_to_pcm(payload, input_rate, self._sample_rate, self._input_resampler)

            if not deserialized_data:
                return None

            # We package the converted audio into a format the Pipecat pipeline understands and send it to the bot
            return InputAudioRawFrame(audio=deserialized_data, num_channels=1, sample_rate=self._sample_rate)

        # If the caller presses a number key on the phone, we detect and send it into the pipeline instead of speech
        if message.get("event") == "dtmf":
            digit = message.get("dtmf", {}).get("digit") or message.get("digit")
            if not digit:
                return None
            try:
                return InputDTMFFrame(KeypadEntry(digit))
            except ValueError:
                logger.warning(f"Invalid DTMF digit received: {digit}")

        return None