import base64
import json
from loguru import logger
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    InputAudioRawFrame,
    StartFrame,
    StartInterruptionFrame,
)

class WebPCMFrameSerializer(FrameSerializer):
    def __init__(self, stream_id: str):
        self._stream_id = stream_id

    async def setup(self, frame: StartFrame):
        pass

    async def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, StartInterruptionFrame):
            return json.dumps({"event": "clearAudio", "streamId": self._stream_id})

        if isinstance(frame, AudioRawFrame):
            head = frame.audio[:10]
            try:
                import struct
                samples = struct.unpack(f"<{len(frame.audio)//2}h", frame.audio)
                max_amp = max(abs(s) for s in samples) if samples else 0
            except Exception:
                max_amp = -1
            
            logger.info(f"[WebPCM] TX -> len:{len(frame.audio)}, sr:{frame.sample_rate}, amp:{max_amp}, head:{head.hex()}")

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
