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
#
# WHY THIS EXISTS: your InterruptionFrame branch below is correct and was
# already doing the right thing -- I got this wrong in an earlier answer
# (I guessed you'd need StartInterruptionFrame; this codebase's turn
# management goes through broadcast_interruption(), which pushes an actual
# InterruptionFrame, per pipecat's own docs: "pushed to interrupt the
# pipeline... to cancel any in-progress bot output"). So if you're still
# hearing the tail of a cut-off sentence after wiring this up, the
# remaining explanation is a narrower one:
#
# InterruptionFrame is a SystemFrame, and pipecat pushes SystemFrames
# "directly downstream" -- bypassing the normal per-processor queue -- so
# it reaches this serializer as fast as physically possible. A regular
# AudioRawFrame that Sarvam's TTS service had already pulled off its own
# websocket a few milliseconds earlier can still be sitting in that
# service's outgoing queue when the interruption overtakes it, and gets
# pushed through moments later. By the time it reaches serialize() here,
# there's no way to tell "this is a genuine new reply's first chunk" from
# "this is a straggler from the utterance we just cancelled" -- both are
# just AudioRawFrame instances. This is a known category of ordering issue
# in pipecat, not unique to this integration: see
# https://github.com/pipecat-ai/pipecat/issues/1323 for the same
# after-the-interruption-frame reordering failure mode on the input side
# ("frames arriving after StopInterruptionFrame... causes bot to repeat
# itself multiple times").
#
# This is a mitigation, not a proof, and it's a time heuristic because
# there is no generation ID to check instead: a genuinely new bot
# utterance needs a full LLM + TTS round trip, which this call's own
# metrics show taking 300-500ms+ end to end (see the
# tts_time_to_first_audio log lines). Audio arriving inside a much shorter
# window right after a clear is essentially certain to be a straggler.
# Tune this against your own TTFA numbers if you change providers, and
# watch for the "Dropped ... stale TTS straggler" warning below in your
# logs -- if you never see it fire, this isn't actually your bug and the
# real explanation is somewhere else (worth checking whether the SAME
# double-voice symptom reproduces on Vobiz calls, not just the web demo --
# if it doesn't, look for something demo/browser-specific instead).
_POST_CLEAR_DROP_WINDOW_S = 0.2


class WebPCMFrameSerializer(FrameSerializer):
    def __init__(self, stream_id: str):
        self._stream_id = stream_id
        self._drop_audio_until = 0.0

    async def setup(self, frame: StartFrame):
        pass

    async def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, InterruptionFrame):
            self._drop_audio_until = time.monotonic() + _POST_CLEAR_DROP_WINDOW_S

            logger.debug(
                "[WebPCM] TX -> clearAudio stream_id={} guard_window={}s",
                self._stream_id,
                _POST_CLEAR_DROP_WINDOW_S,
            )

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

            # Debug logging only when needed - avoid expensive struct.unpack
            logger.debug(f"[WebPCM] TX -> len:{len(frame.audio)}, sr:{frame.sample_rate}")
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
