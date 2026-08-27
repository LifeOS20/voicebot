import httpx
from typing import AsyncGenerator
from loguru import logger
from pipecat.services.tts_service import TTSService
from pipecat.frames.frames import Frame, TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame, ErrorFrame

class HttpElevenLabsTTSService(TTSService):
    def __init__(self, api_key: str, voice_id: str, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.voice_id = voice_id

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.info(f"ElevenLabs HTTP TTS generating: {text}")
        yield TTSStartedFrame()

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream?output_format=pcm_16000"
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
                            yield TTSAudioRawFrame(audio=chunk, sample_rate=16000, num_channels=1)
        except Exception as e:
            logger.error(f"ElevenLabs HTTP Exception: {e}")
            yield ErrorFrame(error=f"ElevenLabs HTTP Exception: {e}")
        finally:
            yield TTSStoppedFrame()
