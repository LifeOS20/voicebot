"""
1. ServiceFactory reads the config file and automatically creates the correct STT, LLM, and TTS services. 
2. run_bot() builds and runs the full Pipecat pipeline for each phone call. 
Every call gets its own pipeline, conversation history, and event handlers.
"""

import os
from importlib import import_module
from typing import Any
import httpx

from dotenv import load_dotenv
from loguru import logger

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContextFrame
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.frames.frames import (
    EndTaskFrame,
    TTSSpeakFrame,
    FunctionCallResultProperties,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.audio.vad.vad_analyzer import VADParams
from fastapi import WebSocket
import json
import asyncio
from services.vobiz_serializer import VobizFrameSerializer
from services.web_serializer import WebPCMFrameSerializer
from prompt_builder import build_system_prompt
from termination_processor import TerminationProcessor

# Load .env file — override=True means .env values take precedence over system environment variables
load_dotenv(override=True)

# Configure logger to file (rotates daily, keeps 10 days)
logger.add("logs/voicebot_{time:YYYY-MM-DD}.log", rotation="1 day", retention="10 days", level="DEBUG")

STREAM_PROVIDER_CALL_IDS: dict[str, str] = {}

# Global VAD Instance
VAD_ANALYZER = None

class ServiceFactory:
    # Dynamic factory for creating AI services from config
    @staticmethod
    def _import_class(class_path: str) -> Any:
        module_path, class_name = class_path.rsplit(".", 1)
        module = import_module(module_path)
        return getattr(module, class_name)

    @classmethod
    def create(cls, service_type: str, provider_name: str, config: dict, **dynamic_kwargs) -> Any:
        # Creates and returns a ready-to-use STT, LLM, or TTS service using the chosen provider from the config file.
        # Look up the provider in the config registry
        providers = config.get("providers", {}).get(service_type, {})
        provider_config = providers.get(provider_name)

        if not provider_config:
            available = ", ".join(sorted(providers.keys()))
            raise ValueError(
                f"Provider '{provider_name}' not found for {service_type}. "
                f"Available: {available}"
            )

        # Get class path for dynamic import
        class_path = provider_config.get("class_path")
        if not class_path:
            raise ValueError(f"Missing 'class_path' for {service_type}:{provider_name}")

        # Attempt the dynamic import — if the package isn't installed, it will fail
        try:
            service_class = cls._import_class(class_path)
        except (ImportError, AttributeError) as e:
            raise RuntimeError(
                f"Failed to import {class_path}. "
                f"Ensure the package is installed. Error: {e}"
            )

        # kwargs dict for the service constructor
        kwargs = {}

        # Fetch the API key from environment variables.
        api_key_env = provider_config.get("api_key_env")
        if api_key_env:
            api_key = os.getenv(api_key_env)
            if not api_key:
                raise ValueError(f"Missing environment variable: {api_key_env}")
            kwargs["api_key"] = api_key

        # Merge any additional params (model name, base_url, voice_id, etc.)
        params = provider_config.get("params", {})
        kwargs.update(params)
        kwargs.update(dynamic_kwargs)

        # creating a service object
        logger.info(f"Creating {service_type} service: {provider_name}")
        return service_class(**kwargs)


def _prune_conversation_history(messages: list, max_messages: int) -> None:
    """
    Keeps the chat history from getting too long for the LLM. It always keeps the first message (the system prompt), 
    removes the oldest conversation turns when the list gets too big, and keeps the most recent messages so the conversation still makes sense. 
    This cleanup runs after each turn to keep the history within safe limits.
    """
    if len(messages) <= max_messages:
        return  # Nothing to prune

    # Keep system prompt (index 0) + the most recent (max_messages - 1) messages
    system_prompt = messages[0]
    recent_messages = messages[-(max_messages - 1):]
    messages.clear()
    messages.append(system_prompt)
    messages.extend(recent_messages)

    logger.debug(f"Pruned conversation history to {len(messages)} messages")


async def run_bot(
    websocket: WebSocket,
    call_type: str,
    config: dict,
    stream_id: str | None = None,
    campaign_data: dict | None = None,
    call_id: str | None = None,
):
    """
    Builds and runs the bot for a single phone call.
    It runs once for each WebSocket connection, meaning every call gets a fresh conversation history so calls never mix with each other.
    """
    provider_call_id = campaign_data.get("provider_call_id") if campaign_data else None

    # If stream_id is missing, parse it from the initial WebSocket message
    if not stream_id:
        try:
            # Wait for the "start" message from Vobiz
            # We might receive multiple messages (e.g., "connected", "ringing"?, then "start")
            # We loop until we find a message with a valid stream_id.
            for attempt in range(5):  # Try 5 messages max as a safeguard
                message = await websocket.receive_text()
                logger.debug(f"WebSocket message {attempt+1}: {message[:200]}...")
                
                data = json.loads(message)
                
                # Check for streamId directly or in 'start' object
                candidate_id = (data.get("streamId") or data.get("start", {}).get("streamId"))
                candidate_call_id = (data.get("callId") or data.get("start", {}).get("callId"))
                if candidate_call_id:
                    provider_call_id = candidate_call_id
                
                if candidate_id:
                    stream_id = candidate_id
                    logger.info(f"[{stream_id}] Found stream_id in message {attempt+1}")
                    break
            
            if not stream_id:
                logger.warning("Could not find stream_id in initial messages, defaulting to 'unknown'")
                stream_id = "unknown"
        except Exception as e:
            logger.error(f"Failed to parse initial WebSocket message: {e}")
            stream_id = "unknown_error"

    logger.info(f"[{stream_id}] Starting {call_type} call (Call ID: {call_id})")

# provider_call_id - Vobiz creates it on THEIR side when they actually initiate the phone call
    if stream_id and provider_call_id:
        STREAM_PROVIDER_CALL_IDS[stream_id] = provider_call_id
    elif stream_id and not provider_call_id:
        provider_call_id = STREAM_PROVIDER_CALL_IDS.get(stream_id)
        if provider_call_id:
            logger.info(f"[{stream_id}] Reusing cached provider callId for hangup fallback")

    async def _force_provider_hangup(trigger: str) -> None:
        """
        Manually tells the telephony provider (Vobiz) to end the call immediately.
        This is a backup in case the provider doesn't detect the hangup automatically.
        """
        # 1. We need the specific Call ID from Vobiz to know which call to stop.
        if not provider_call_id:
            logger.warning(f"[{stream_id}] Cannot force hangup ({trigger}): missing provider callId")
            return

        # 2. Get the API keys needed to talk to Vobiz.
        auth_id = os.getenv("VOBIZ_AUTH_ID")
        auth_token = os.getenv("VOBIZ_AUTH_TOKEN")
        if not auth_id or not auth_token:
            logger.warning(f"[{stream_id}] Cannot force hangup ({trigger}): missing VOBIZ_AUTH_ID/VOBIZ_AUTH_TOKEN")
            return
        
        # 3. Prepare the API request to delete (end) the active call.
        hangup_url = f"https://api.vobiz.ai/api/v1/Account/{auth_id}/Call/{provider_call_id}/"
        headers = {
            "X-Auth-ID": auth_id,
            "X-Auth-Token": auth_token,
            "Content-Type": "application/json",
        }
        
        # 4. Send the request to Vobiz and check if they received it successfully.
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.delete(hangup_url, headers=headers)
                if response.status_code in (200, 201, 202, 204):
                    logger.info(f"[{stream_id}] Provider hangup succeeded via API ({trigger})")
                else:
                    logger.warning(f"[{stream_id}] Provider hangup failed: {response.status_code} {response.text}")
        except Exception as e:
            # If the request fails (e.g. network error), just log it and continue shutting down.
            logger.warning(f"[{stream_id}] Provider hangup exception ({trigger}): {e}")

    # Per-call file logging
    # Create a unique log file for this specific call to isolate logs
    call_log_file = f"logs/call_{call_id}_{stream_id}.log"
    handler_id = logger.add(call_log_file, level="DEBUG")

    # Set up serializer based on call type
    if call_type == "web":
        serializer = WebPCMFrameSerializer(stream_id=stream_id)
    else:
        serializer = VobizFrameSerializer(stream_id=stream_id, sample_rate=8000)

    # Configure VAD
    # Reuse global VAD analyzer if available to save load time (approx 1-2s)
    global VAD_ANALYZER
    if VAD_ANALYZER is None and config.get("vad", {}).get("enabled", True):
        vad_config = config.get("vad", {})
        logger.info("Initializing Global Silero VAD...")
        VAD_ANALYZER = SileroVADAnalyzer(
            params=VADParams(
                start_secs=vad_config.get("min_speech_duration", 0.25),
                stop_secs=vad_config.get("silence_timeout", 0.8),
                confidence=vad_config.get("start_threshold", 0.5)
            )
        )
    
    vad_analyzer = VAD_ANALYZER
    active_providers = config["active_providers"]

    # Deepgram closes idle websockets when no audio/text arrives for a while.
    # Passing audio through VAD keeps the socket healthy during pauses.
    use_vad_passthrough = active_providers.get("stt") == "deepgram"

    # Create Transport
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_enabled=True,
            vad_analyzer=vad_analyzer,
            vad_audio_passthrough=use_vad_passthrough,
            serializer=serializer,
        ),
    )

    # Services are created per-call to ensure thread safety and isolation
    try:
        sample_rate = 16000 if call_type == "web" else config["audio"]["sample_rate"]
        stt = ServiceFactory.create("stt", active_providers["stt"], config, sample_rate=sample_rate)
        llm = ServiceFactory.create("llm", active_providers["llm"], config)
        tts = ServiceFactory.create("tts", active_providers["tts"], config, sample_rate=sample_rate)
    except Exception as e:
        logger.error(f"[{stream_id}] Failed to create services: {e}")
        return

    # Shared shutdown state to coordinate between tool calls and frame processor
    shutdown_state = {"active": False}

    end_call_pending = {"active": False}

    # If the regex-based TerminationProcessor already caught a goodbye (shutdown_state is true), we skip injection. 
    # Otherwise, inject a TTS goodbye so the user always hears one.
    async def end_call_handler(function_name, tool_call_id, args, llm, context, result_callback):
        logger.info(f"[{stream_id}] LLM invoked end_call. shutdown_state={shutdown_state['active']}")
        end_call_pending["active"] = True

        if not shutdown_state["active"]:
            # Regex didn't catch a goodbye yet — inject one via TTS
            logger.info(f"[{stream_id}] No prior goodbye detected. Injecting TTS goodbye.")
            await llm.push_frame(TTSSpeakFrame(text="Thank you for your time. Goodbye."), FrameDirection.DOWNSTREAM)

        # Fallback: forcefully hang up after 8s if bot-stop handler doesn't catch it
        async def _fallback_hangup():
            await asyncio.sleep(8.0)
            if not shutdown_state["active"]:
                logger.info(f"[{stream_id}] end_call fallback hangup triggered.")
                shutdown_state["active"] = True
                await _force_provider_hangup("llm_end_call_fallback")
                await llm.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)

        asyncio.create_task(_fallback_hangup())

        await result_callback(
            {"status": "ending"},
            properties=FunctionCallResultProperties(run_llm=False),
        )

    llm.register_function("end_call", end_call_handler)


    # Build system prompt (voice rules + campaign/base prompt + termination rules)
    system_prompt = build_system_prompt(call_type, config, campaign_data)
    logger.info(f"[{stream_id}] System prompt built ({len(system_prompt)} chars)")

    # Each call gets its own message history
    # Starts with just the system prompt and grows as the conversation progresses.
    messages = [{"role": "system", "content": system_prompt}]

    # Extract tools from LLM provider config so they are sent with every API request
    llm_provider = active_providers["llm"]
    llm_tools = config.get("providers", {}).get("llm", {}).get(llm_provider, {}).get("params", {}).get("tools", [])
    if llm_tools:
        logger.info(f"[{stream_id}] Registering {len(llm_tools)} tool(s) with LLM context")

    #OpenAILLMContext stores the conversation history 
    #and lets Pipecat automatically add what the user says and what the bot replies.
    context = OpenAILLMContext(messages, tools=llm_tools if llm_tools else None)
    context_aggregator = llm.create_context_aggregator(context)

    # Instantiate the external TerminationProcessor with required state and config-driven patterns
    tp_config = config.get("termination_processor", {})
    termination_processor = TerminationProcessor(
        shutdown_state=shutdown_state,
        end_call_pending=end_call_pending,
        stream_id=stream_id,
        force_hangup_callback=_force_provider_hangup,
        silent_patterns=tp_config.get("silent_patterns", []),
        spoken_patterns=tp_config.get("spoken_patterns", []),
    )

    # Assemble the Pipecat pipeline
    # TTS starts speaking while the LLM is still generating the rest. This keeps Time-to-First-Byte (TTFB) extremely low.
    pipeline = Pipeline([
        transport.input(), # Receives raw audio from Vobiz WebSocket
        stt,  # ASR
        context_aggregator.user(), # Saves users text to conversation history
        llm, # Generates a response based on full context
        termination_processor,
        tts, # Convert the text to audio
        transport.output(), # Send audio back to vobiz - caller hears it
        context_aggregator.assistant() # Save bot's response to conversation history
    ])

    # Pipeline task configuration
    task = PipelineTask(pipeline,params=PipelineParams(
            # Audio sample rates - 8000 Hz is standard telephony, 16000 Hz for web clients
            audio_in_sample_rate=16000 if call_type == "web" else config["audio"]["sample_rate"],
            audio_out_sample_rate=16000 if call_type == "web" else config["audio"]["sample_rate"],

           # If the user starts speaking while the bot is talking, the bot stops speaking, clears any pending audio, and starts listening
            allow_interruptions=config["audio"]["enable_interruptions"],

            # Useful for identifying bottlenecks 
            enable_metrics=config["metrics"]["enable_metrics"],
            enable_usage_metrics=config["metrics"]["enable_usage_metrics"],
        ),
    )

    # Event handlers
    # We keep the system prompt + last 19 exchanges. So that token limit doesnt exceed
    max_messages = config.get("max_conversation_messages", 20)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        """
        When the call audio starts, the bot sends a greeting first to avoid silence, 
        Gave it as a system message so the LLM knows to speak before the user says anything.
        """
        logger.info(f"[{stream_id}] Call connected ({call_type})")

        # Use campaign greeting if provided, otherwise fall back to config
        if campaign_data and campaign_data.get("greeting"):
            customer_name = campaign_data.get("customer_name", "there")
            greeting = campaign_data["greeting"].replace("{customer_name}", customer_name)
        else:
            greeting_key = f"greeting_{call_type}"
            greeting = config.get(greeting_key, "Say a brief greeting.")

        logger.info(f"[{stream_id}] Sending greeting: {greeting[:100]}...")
        messages.append({"role": "user", "content": greeting})

        # LLM processes the greeting prompt and generates the spoken greeting via TTS.
        await task.queue_frames([OpenAILLMContextFrame(context)])

        # Safety timeout: hang up after 5 minutes regardless
        async def safety_timeout():
            await asyncio.sleep(300)  # 5 minutes
            logger.warning(f"[{stream_id}] Call exceeded 5 min, forcing hangup")
            await _force_provider_hangup("hard_timeout")
            await task.cancel()
        
        asyncio.create_task(safety_timeout())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        # When the call ends, the pipeline task is cancelled and the bot shuts down for that call.
        logger.info(f"[{stream_id}] Call disconnected ({call_type})")
        await task.cancel()

    # Run the pipeline
    runner = PipelineRunner(handle_sigint=False) #server does shutdown

    try:
        await runner.run(task)
    except Exception as e:
        # If the pipeline crashes during a call, log the error, keep the server running, and the caller will eventually disconnect after hearing silence.
        logger.error(f"[{stream_id}] Pipeline error during {call_type} call: {e}")
    finally:
        # Remove the per-call log handler
        logger.remove(handler_id)
        
        # Conversation history is cleared after the call ends so old messages do not affect future calls
        _prune_conversation_history(messages, max_messages)
        logger.info(f"[{stream_id}] Pipeline finished ({call_type})")
