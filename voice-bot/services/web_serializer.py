import time
import base64
import json
from loguru import logger
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    InputAudioRawFrame,
    StartFrame,
    InterruptionFrame,
)

# How long after sending clearAudio to keep dropping outgoing audio frames
# instead of forwarding them as playAudio.
# Increased to 1.0s to match Vobiz serializer and account for network latency + pacing.
_POST_CLEAR_DROP_WINDOW_S = 1.0


class WebPCMFrameSerializer(FrameSerializer):
    def __init__(self, stream_id: str):
        self._stream_id = stream_id
        self._drop_audio_until = 0.0

    async def setup(self, frame: StartFrame):
        pass

    def on_interruption(self) -> None:
        """Call when barge-in detected - activates drop window."""
        self._drop_audio_until = time.monotonic() + _POST_CLEAR_DROP_WINDOW_S
        logger.debug(
            "[WebPCM] TX -> clearAudio stream_id={} guard_window={}s",
            self._stream_id,
            _POST_CLEAR_DROP_WINDOW_S,
        )

    def next_generation(self) -> int:
        """Call when a new user turn starts - clears any pending drop window."""
        self._drop_audio_until = 0.0
        logger.debug(
            "[WebPCMSerializer] New generation started, drop window cleared stream_id={}",
            self._stream_id,
        )
        return 1

    async def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, InterruptionFrame):
            self.on_interruption()
            return json.dumps({"event": "clearAudio", "streamId": self._stream_id})

        if isinstance(frame, AudioRawFrame):
            if time.monotonic() < self._drop_audio_until:
                logger.warning(
                    "[WebPCM] Dropped {} bytes of audio inside the "
                    "post-interruption guard window (stale TTS straggler) "
                    "stream_id={}",
                    len(frame.audio),
                    self._stream_id,
                )
                return None

            payload = base64.b64encode(frame.audio).decode("utf-8")
            return json.dumps(
                {
                    "event": "playAudio",
                    "streamId": self._stream_id,
                    "media": {
                        "contentType": "audio/x-l16",
                        "sampleRate": frame.sample_rate,
                        "payload": payload,
                    },
                }
            )

        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        try:
            message = json.loads(data)
        except json.JSONDecodeError:
            return None
        if message.get("event") == "media":
            media = message.get("media", {})
            payload_base64 = media.get("payload")
            if not payload_base64:
                return None

            payload = base64.b64decode(payload_base64)
            input_rate = int(media.get("sampleRate", 16000))
            return InputAudioRawFrame(audio=payload, num_channels=1, sample_rate=input_rate)
        return None