import httpx
from typing import AsyncGenerator
from loguru import logger
from pipecat.services.tts_service import TTSService
from pipecat.frames.frames import Frame, TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame, ErrorFrame

class HttpElevenLabsTTSService(TTSService):
    def __init__(self, api_key: str, voice_id: str, sample_rate: int = 16000, **kwargs):
        # FIXED: this used to hardcode 16000 in both the API request and the
        # emitted audio frame regardless of what sample rate the call
        # actually needed. Telephony calls run at 8kHz (to match the Vobiz
        # mu-law serializer) — requesting 16kHz audio and labeling it as
        # 16kHz while the rest of the pipeline expects 8kHz would produce
        # garbled, wrong-speed audio on a real phone call. This provider
        # isn't active in config.yaml today (murf is), but it's still
        # selectable, so this landmine needed defusing regardless.
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.api_key = api_key
        self.voice_id = voice_id

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.info(f"ElevenLabs HTTP TTS generating: {text}")
        yield TTSStartedFrame()

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream?output_format=pcm_{self.sample_rate}"
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "text": text,
            "model_id": "eleven_turbo_v2_5"
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                async with client.stream("POST", url, headers=headers, json=payload) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        logger.error(f"ElevenLabs API Error [{response.status_code}]: {error_text.decode()}")
                        yield ErrorFrame(error=f"ElevenLabs HTTP failed: {response.status_code}")
                        yield TTSStoppedFrame()
                        return

                    async for chunk in response.aiter_bytes(chunk_size=1280):
                        if chunk:
                            yield TTSAudioRawFrame(audio=chunk, sample_rate=self.sample_rate, num_channels=1)
        except Exception as e:
            logger.error(f"ElevenLabs HTTP Exception: {e}")
            yield ErrorFrame(error=f"ElevenLabs HTTP Exception: {e}")
        finally:
            yield TTSStoppedFrame()
