import asyncio
import os
import yaml
from loguru import logger
from dotenv import load_dotenv

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.frames.frames import LLMContextFrame, TTSSpeakFrame, FunctionCallResultProperties
from pipecat.services.llm_service import FunctionCallParams
from pipecat.processors.frame_processor import FrameDirection
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams

# Local transports
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

from bot import ServiceFactory
from prompt_builder import build_system_prompt

load_dotenv(override=True)

async def main():
    logger.info("Initializing local test bot...")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    vad_config = config.get("vad", {})
    vad_analyzer = SileroVADAnalyzer(
        params=VADParams(
            start_secs=vad_config.get("min_speech_duration", 0.25),
            stop_secs=vad_config.get("silence_timeout", 0.8),
            confidence=vad_config.get("start_threshold", 0.5)
        )
    )

    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=config["audio"]["sample_rate"],
            vad_enabled=True,
            vad_analyzer=vad_analyzer,
        )
    )

    active_providers = config["active_providers"]
    try:
        stt = ServiceFactory.create("stt", active_providers["stt"], config)
        llm = ServiceFactory.create("llm", active_providers["llm"], config)
        tts = ServiceFactory.create("tts", active_providers["tts"], config)
    except Exception as e:
        logger.error(f"Failed to create services: {e}")
        return

    # Mock end_call handler
    shutdown_state = {"active": False}

    async def end_call_handler(params: FunctionCallParams):
        logger.info("LLM invoked end_call. Ending local test.")
        shutdown_state["active"] = True
        await params.llm.push_frame(TTSSpeakFrame(text="Thank you for your time. Goodbye."), FrameDirection.DOWNSTREAM)

        async def _fallback_hangup():
            await asyncio.sleep(3.0)
            await task.cancel()

        asyncio.create_task(_fallback_hangup())

        await params.result_callback(
            {"status": "ending"},
            properties=FunctionCallResultProperties(run_llm=False),
        )

    llm.register_function("end_call", end_call_handler)

    system_prompt = build_system_prompt("inbound", config, None)
    messages = [{"role": "system", "content": system_prompt}]

    llm_provider = active_providers["llm"]
    llm_tools_raw = config.get("providers", {}).get("llm", {}).get(llm_provider, {}).get("params", {}).get("tools", [])
    llm_tools = [
        FunctionSchema(
            name=raw["function"]["name"],
            description=raw["function"].get("description", ""),
            properties=raw["function"].get("parameters", {}).get("properties", {}),
            required=raw["function"].get("parameters", {}).get("required", []),
        )
        for raw in llm_tools_raw
    ]

    context = LLMContext(messages, tools=llm_tools)
    context_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline([
        transport.input(),
        stt,
        context_aggregator.user(),
        llm,
        tts,
        transport.output(),
        context_aggregator.assistant()
    ])

    task = PipelineTask(pipeline, params=PipelineParams(
        audio_in_sample_rate=16000,
        audio_out_sample_rate=config["audio"]["sample_rate"],
        allow_interruptions=config["audio"]["enable_interruptions"],
    ))

    logger.info("Local script starting. Greeting the user...")
    greeting = config.get("greeting_inbound", "Hello! How can I help you today?")
    messages.append({"role": "user", "content": greeting})
    await task.queue_frames([LLMContextFrame(context=context)])

    runner = PipelineRunner(handle_sigint=True)

    try:
        await runner.run(task)
    except KeyboardInterrupt:
        logger.info("Local test interrupted by user.")
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
    finally:
        logger.info("Local test finished.")

if __name__ == "__main__":
    asyncio.run(main())
