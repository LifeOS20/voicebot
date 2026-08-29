import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

import aiohttp


async def main():
    url = "https://in.api.murf.ai/v1/speech/stream"

    headers = {
        "api-key": os.environ["MURF_API_KEY"],
        "Accept": "application/octet-stream",
        "Content-Type": "application/json",
    }

    payload = {
        "text": "Hello, this is a test of the Murf voice.",
        "voiceId": "Anisha",
        "model": "falcon-2",
        "locale": "en-IN",
        "channelType": "MONO",
        "format": "PCM",
        "sampleRate": 8000,
    }

    timeout = aiohttp.ClientTimeout(total=30)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            url,
            headers=headers,
            json=payload,
        ) as response:
            print("status:", response.status)
            print("content-type:", response.headers.get("Content-Type"))

            body = await response.read()
            print("bytes received:", len(body))

            if response.status != 200:
                print(body[:1000].decode(errors="replace"))
                return

            with open("murf_test.pcm", "wb") as f:
                f.write(body)

            print("saved: murf_test.pcm")


asyncio.run(main())