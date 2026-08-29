import asyncio
import os

import aiohttp
from dotenv import load_dotenv

load_dotenv()

from services.murf_http import MurfFalcon2TTSService


async def main():
    session = aiohttp.ClientSession()

    try:
        tts = MurfFalcon2TTSService(
            api_key=os.environ["MURF_API_KEY"],
            sample_rate=8000,
            aiohttp_session=session,
        )

        print("sample_rate:", tts.sample_rate)
        print("settings:", tts._settings)
        print("model:", tts._model)
        print("voice:", tts._voice_id)
        print("locale:", tts._locale)
    finally:
        await session.close()


asyncio.run(main())