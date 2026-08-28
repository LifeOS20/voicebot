"""
Production voicebot orchestration.

The bot deliberately does NOT contain a hand-written language classifier.

Language decisions come from:
1. multilingual STT metadata for automatic detection;
2. the LLM's explicit set_conversation_language tool for intentional changes.

LanguageState is per-call. TTS language changes are performed through a
provider adapter's public set_language() method.
"""

from __future__ import annotations

import asyncio
import json
import os
from importlib import import_module
from typing import Any, Optional

import aiohttp
import httpx
from dotenv import load_dotenv
from fastapi import WebSocket
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    EndTaskFrame,
    FunctionCallParams,
    FunctionCallResultProperties,
    Frame,
    TranscriptionFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

from language_state import (
    LANGUAGE_LOCALES,
    LanguageCode,
    LanguageState,
    normalize_language,
)
from prompt_builder import build_system_prompt
from services.vobiz_serializer import VobizFrameSerializer
from services.web_serializer import WebPCMFrameSerializer
from termination_processor import TerminationProcessor

load_dotenv(override=True)

os.makedirs("logs", exist_ok=True)

logger.add(
    "logs/voicebot_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="10 days",
    level="DEBUG",
)

STREAM_PROVIDER_CALL_IDS: dict[str, str] = {}


class LanguageObserver(FrameProcessor):
    """Observe STT language metadata for one call and update TTS on switches."""

    def __init__(
        self,
        *,
        stream_id: str,
        language_state: LanguageState,
        tts: Any,
        tts_provider: str,
    ) -> None:
        super().__init__()
        self._stream_id = stream_id
        self._language_state = language_state
        self._tts = tts
        self._tts_provider = tts_provider

    async def process_frame(
        self,
        frame: Frame,
        direction: FrameDirection,
    ) -> None:
        await super().process_frame(frame, direction)

        if (
            direction == FrameDirection.DOWNSTREAM
            and isinstance(frame, TranscriptionFrame)
        ):
            language, probability = self._extract_language(
                frame
            )

            if language:
                old_language = (
                    self._language_state.current_language
                )

                new_language, switched = (
                    self._language_state.observe_stt(
                        language,
                        probability,
                    )
                )

                logger.info(
                    "[{}] language observation old={} detected={} "
                    "confidence={} current={} switched={} reason={}",
                    self._stream_id,
                    old_language,
                    language,
                    probability,
                    new_language,
                    switched,
                    self._language_state.switch_reason,
                )

                if switched:
                    await self._update_tts_language(
                        new_language
                    )

        await self.push_frame(
            frame,
            direction,
        )

    @staticmethod
    def _extract_language(
        frame: TranscriptionFrame,
    ) -> tuple[Optional[str], Optional[float]]:
        """Read normalized language metadata from provider output.

        Sarvam's realtime result may expose metadata under frame.result.
        Keep the provider-specific extraction isolated here.
        """
        raw_language = getattr(
            frame,
            "language",
            None,
        )

        raw_probability = getattr(
            frame,
            "language_probability",
            None,
        )

        language = normalize_language(
            getattr(
                raw_language,
                "value",
                raw_language,
            )
        )

        if language:
            return language, _normalize_probability(
                raw_probability
            )

        result = getattr(
            frame,
            "result",
            None,
        )

        if isinstance(result, dict):
            # Sarvam realtime frame metadata can be nested under data.
            data = result.get(
                "data"
            )

            containers = [
                result,
                data if isinstance(data, dict) else {},
            ]

            for container in containers:
                raw_language = (
                    container.get("language_code")
                    or container.get("language")
                    or container.get("detected_language")
                )

                raw_probability = (
                    container.get("language_probability")
                    or container.get("language_confidence")
                    or container.get("confidence")
                )

                language = normalize_language(
                    raw_language
                )

                if language:
                    return (
                        language,
                        _normalize_probability(
                            raw_probability
                        ),
                    )

        return None, None

    async def _update_tts_language(
        self,
        language: LanguageCode,
    ) -> None:
        setter = getattr(
            self._tts,
            "set_language",
            None,
        )

        if not callable(setter):
            logger.error(
                "[{}] TTS provider '{}' does not expose "
                "set_language(); cannot safely switch language",
                self._stream_id,
                self._tts_provider,
            )
            return

        await setter(
            LANGUAGE_LOCALES[language]
        )


def _normalize_probability(
    value: object,
) -> Optional[float]:
    try:
        result = float(value)

        if result > 1:
            result /= 100.0

        return max(
            0.0,
            min(1.0, result),
        )
    except (TypeError, ValueError):
        return None


class ServiceFactory:
    """Configuration-driven provider construction."""

    @staticmethod
    def _import_class(
        class_path: str,
    ) -> Any:
        module_path, class_name = class_path.rsplit(
            ".",
            1,
        )

        module = import_module(
            module_path
        )

        return getattr(
            module,
            class_name,
        )

    @classmethod
    def create(
        cls,
        service_type: str,
        provider_name: str,
        config: dict,
        *,
        aiohttp_session: aiohttp.ClientSession | None = None,
        **dynamic_kwargs: Any,
    ) -> Any:

        registry = (
            config
            .get("providers", {})
            .get(service_type, {})
        )

        provider_config = registry.get(
            provider_name
        )

        if not provider_config:
            raise ValueError(
                f"Unknown {service_type} provider: "
                f"{provider_name}"
            )

        class_path = provider_config.get(
            "class_path"
        )

        if not class_path:
            raise ValueError(
                f"Missing class_path for "
                f"{service_type}:{provider_name}"
            )

        try:
            service_class = cls._import_class(
                class_path
            )
        except (
            ImportError,
            AttributeError,
        ) as exc:
            raise RuntimeError(
                f"Cannot import {class_path}: {exc}"
            ) from exc

        kwargs: dict[str, Any] = {}

        api_key_env = provider_config.get(
            "api_key_env"
        )

        if api_key_env:
            api_key = os.getenv(
                api_key_env
            )

            if not api_key:
                raise ValueError(
                    f"Missing environment variable: "
                    f"{api_key_env}"
                )

            kwargs["api_key"] = api_key

        if provider_config.get(
            "_needs_aiohttp",
            False,
        ):
            if aiohttp_session is None:
                raise RuntimeError(
                    f"{service_type}:{provider_name} "
                    "requires aiohttp_session"
                )

            kwargs["aiohttp_session"] = (
                aiohttp_session
            )

        params = dict(
            provider_config.get(
                "params",
                {},
            )
        )

        # The custom Murf adapter intentionally accepts normal kwargs.
        # Current Pipecat services may expose Settings; support both
        # without putting provider branches in run_bot().
        if "MurfFalcon2TTSService" in class_path:
            kwargs.update(params)
        else:
            settings_cls = getattr(
                service_class,
                "Settings",
                None,
            )

            if settings_cls:
                kwargs["settings"] = (
                    settings_cls(**params)
                )
            else:
                input_params_cls = getattr(
                    service_class,
                    "InputParams",
                    None,
                )

                if input_params_cls:
                    kwargs["params"] = (
                        input_params_cls(
                            **params
                        )
                    )
                else:
                    kwargs.update(params)

        kwargs.update(
            dynamic_kwargs
        )

        logger.info(
            "Creating {} provider={} class={}",
            service_type,
            provider_name,
            class_path,
        )

        return service_class(
            **kwargs
        )


def _prune_history(
    messages: list,
    max_messages: int,
) -> None:
    if len(messages) <= max_messages:
        return

    system_message = messages[0]
    messages[:] = [
        system_message,
        *messages[
            -(max_messages - 1):
        ],
    ]


class _HistoryPruner(FrameProcessor):
    """Keeps the LLM context bounded so per-turn token cost doesn't grow
    across a long call.

    FIXED: `_prune_history()` above, and `max_conversation_messages: 20` in
    config.yaml, both existed already — but nothing ever called
    `_prune_history()`. Every turn of a long conversation was re-sending the
    ENTIRE, ever-growing message history to the LLM, so a 15-minute call
    could cost several times more in input tokens by the end than the start.
    This processor runs the existing pruning logic after each turn.
    """

    def __init__(
        self,
        context: OpenAILLMContext,
        max_messages: int,
        stream_id: str,
    ) -> None:
        super().__init__()
        self._context = context
        self._max_messages = max_messages
        self._stream_id = stream_id

    async def process_frame(
        self,
        frame: Frame,
        direction: FrameDirection,
    ) -> None:
        await super().process_frame(frame, direction)

        if direction == FrameDirection.DOWNSTREAM:
            before = len(self._context.messages)
            _prune_history(self._context.messages, self._max_messages)
            after = len(self._context.messages)

            if after < before:
                logger.debug(
                    "[{}] Pruned conversation history {} -> {} messages",
                    self._stream_id,
                    before,
                    after,
                )

        await self.push_frame(frame, direction)


async def run_bot(
    websocket: WebSocket,
    call_type: str,
    config: dict,
    stream_id: str | None = None,
    campaign_data: dict | None = None,
    call_id: str | None = None,
):
    provider_call_id = (
        campaign_data.get(
            "provider_call_id"
        )
        if campaign_data
        else None
    )

    # --------------------------------------------------------
    # Vobiz handshake
    # --------------------------------------------------------
    if not stream_id:
        try:
            for _ in range(5):
                message = await websocket.receive_text()
                data = json.loads(message)

                candidate_stream_id = (
                    data.get("streamId")
                    or data.get(
                        "start",
                        {},
                    ).get("streamId")
                )

                candidate_call_id = (
                    data.get("callId")
                    or data.get(
                        "start",
                        {},
                    ).get("callId")
                )

                if candidate_call_id:
                    provider_call_id = (
                        candidate_call_id
                    )

                if candidate_stream_id:
                    stream_id = (
                        candidate_stream_id
                    )
                    break

            if not stream_id:
                stream_id = "unknown"

        except Exception:
            logger.exception(
                "Failed to parse initial Vobiz messages"
            )
            stream_id = "unknown_error"

    call_label = (
        call_id
        or stream_id
        or "no-call-id"
    )

    if stream_id and provider_call_id:
        STREAM_PROVIDER_CALL_IDS[
            stream_id
        ] = provider_call_id

    elif stream_id:
        provider_call_id = (
            STREAM_PROVIDER_CALL_IDS.get(
                stream_id
            )
        )

    logger.info(
        "[{}] Starting {} call call_id={} providers={}",
        stream_id,
        call_type,
        call_label,
        config.get(
            "active_providers",
            {},
        ),
    )

    # --------------------------------------------------------
    # Provider hangup fallback
    # --------------------------------------------------------
    async def force_provider_hangup(
        trigger: str,
    ) -> None:

        if not provider_call_id:
            logger.warning(
                "[{}] Missing Vobiz call ID ({})",
                stream_id,
                trigger,
            )
            return

        auth_id = os.getenv(
            "VOBIZ_AUTH_ID"
        )
        auth_token = os.getenv(
            "VOBIZ_AUTH_TOKEN"
        )

        if not auth_id or not auth_token:
            logger.warning(
                "[{}] Missing Vobiz auth credentials ({})",
                stream_id,
                trigger,
            )
            return

        url = (
            f"https://api.vobiz.ai/api/v1/"
            f"Account/{auth_id}/Call/"
            f"{provider_call_id}/"
        )

        try:
            async with httpx.AsyncClient(
                timeout=5.0
            ) as client:
                response = await client.delete(
                    url,
                    headers={
                        "X-Auth-ID": auth_id,
                        "X-Auth-Token": auth_token,
                        "Content-Type": "application/json",
                    },
                )

            if response.status_code in {
                200,
                201,
                202,
                204,
            }:
                logger.info(
                    "[{}] Vobiz hangup succeeded ({})",
                    stream_id,
                    trigger,
                )
            else:
                logger.warning(
                    "[{}] Vobiz hangup failed status={} ({})",
                    stream_id,
                    response.status_code,
                    trigger,
                )

        except Exception:
            logger.exception(
                "[{}] Vobiz hangup request failed ({})",
                stream_id,
                trigger,
            )

    # --------------------------------------------------------
    # Per-call resources
    # --------------------------------------------------------
    log_handler = logger.add(
        f"logs/call_{call_label}_{stream_id}.log",
        level="DEBUG",
    )

    aiohttp_session = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(
            total=60,
            connect=10,
        ),
        connector=aiohttp.TCPConnector(
            limit=20,
            ttl_dns_cache=300,
        ),
    )

    try:
        serializer = (
            WebPCMFrameSerializer(
                stream_id=stream_id
            )
            if call_type == "web"
            else VobizFrameSerializer(
                stream_id=stream_id,
                sample_rate=8000,
            )
        )

        vad_analyzer = None

        if config.get(
            "vad",
            {},
        ).get("enabled", True):

            vad_config = config["vad"]

            vad_analyzer = SileroVADAnalyzer(
                params=VADParams(
                    start_secs=vad_config.get(
                        "min_speech_duration",
                        0.25,
                    ),
                    stop_secs=vad_config.get(
                        "silence_timeout",
                        0.8,
                    ),
                    confidence=vad_config.get(
                        "start_threshold",
                        0.55,
                    ),
                )
            )

        active = config[
            "active_providers"
        ]

        transport = FastAPIWebsocketTransport(
            websocket=websocket,
            params=FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                add_wav_header=False,
                vad_enabled=True,
                vad_analyzer=vad_analyzer,
                vad_audio_passthrough=(
                    active.get("stt")
                    == "deepgram"
                ),
                serializer=serializer,
            ),
        )

        sample_rate = (
            16000
            if call_type == "web"
            else config["audio"]["sample_rate"]
        )

        stt = ServiceFactory.create(
            "stt",
            active["stt"],
            config,
            sample_rate=sample_rate,
        )

        llm = ServiceFactory.create(
            "llm",
            active["llm"],
            config,
        )

        tts = ServiceFactory.create(
            "tts",
            active["tts"],
            config,
            sample_rate=sample_rate,
            aiohttp_session=aiohttp_session,
        )

        language_state = LanguageState()

        shutdown_state = {
            "active": False
        }

        end_call_pending = {
            "active": False
        }

        # ----------------------------------------------------
        # LLM tools
        # ----------------------------------------------------
        async def end_call(
            params: FunctionCallParams,
        ):
            logger.info(
                "[{}] end_call invoked language={}",
                stream_id,
                language_state.current_language,
            )

            end_call_pending[
                "active"
            ] = True

            if not shutdown_state[
                "active"
            ]:
                await params.llm.push_frame(
                    TTSSpeakFrame(
                        text=(
                            "Thank you for your time. Goodbye."
                        )
                    ),
                    FrameDirection.DOWNSTREAM,
                )

            async def fallback():
                await asyncio.sleep(8)

                if not shutdown_state[
                    "active"
                ]:
                    shutdown_state[
                        "active"
                    ] = True

                    await force_provider_hangup(
                        "llm_end_call_fallback"
                    )

                    await params.llm.push_frame(
                        EndTaskFrame(),
                        FrameDirection.UPSTREAM,
                    )

            asyncio.create_task(
                fallback()
            )

            await params.result_callback(
                {
                    "status": "ending"
                },
                properties=(
                    FunctionCallResultProperties(
                        run_llm=False
                    )
                ),
            )

        async def set_conversation_language(
            params: FunctionCallParams,
        ):
            arguments = dict(
                params.arguments
            )

            language = normalize_language(
                arguments.get(
                    "language"
                )
            )

            reason = str(
                arguments.get(
                    "reason",
                    "caller language preference changed",
                )
            )

            if not language:
                await params.result_callback(
                    {
                        "status": "rejected",
                        "reason": "unsupported language",
                    }
                )
                return

            try:
                new_language, switched = (
                    language_state.set_explicit(
                        language,
                        reason,
                    )
                )

                setter = getattr(
                    tts,
                    "set_language",
                    None,
                )

                if switched:
                    if not callable(setter):
                        raise RuntimeError(
                            f"TTS provider "
                            f"'{active['tts']}' does not "
                            "support runtime language switching"
                        )

                    await setter(
                        LANGUAGE_LOCALES[
                            new_language
                        ]
                    )

                await params.result_callback(
                    {
                        "status": (
                            "changed"
                            if switched
                            else "already_active"
                        ),
                        "language": new_language,
                    }
                )

            except Exception as exc:
                logger.exception(
                    "[{}] Language switch failed",
                    stream_id,
                )

                await params.result_callback(
                    {
                        "status": "failed",
                        "language": (
                            language_state.current_language
                        ),
                        "error": str(exc),
                    }
                )

        llm.register_function(
            "end_call",
            end_call,
            cancel_on_interruption=True,
        )

        llm.register_function(
            "set_conversation_language",
            set_conversation_language,
            cancel_on_interruption=True,
        )

        # ----------------------------------------------------
        # Prompt/context
        # ----------------------------------------------------
        system_prompt = build_system_prompt(
            call_type,
            config,
            campaign_data,
        )

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]

        llm_tools = (
            config
            .get("providers", {})
            .get("llm", {})
            .get(active["llm"], {})
            .get("params", {})
            .get("tools", [])
        )

        context = OpenAILLMContext(
            messages,
            tools=llm_tools or None,
        )

        context_aggregator = (
            llm.create_context_aggregator(
                context
            )
        )

        language_observer = LanguageObserver(
            stream_id=stream_id,
            language_state=language_state,
            tts=tts,
            tts_provider=active["tts"],
        )

        termination_config = config.get(
            "termination_processor",
            {},
        )

        termination_processor = (
            TerminationProcessor(
                shutdown_state=shutdown_state,
                end_call_pending=end_call_pending,
                stream_id=stream_id,
                force_hangup_callback=(
                    force_provider_hangup
                ),
                silent_patterns=termination_config.get(
                    "silent_patterns",
                    [],
                ),
                spoken_patterns=termination_config.get(
                    "spoken_patterns",
                    [],
                ),
            )
        )

        history_pruner = _HistoryPruner(
            context=context,
            max_messages=config.get(
                "max_conversation_messages",
                20,
            ),
            stream_id=stream_id,
        )

        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                language_observer,
                context_aggregator.user(),
                llm,
                termination_processor,
                tts,
                transport.output(),
                context_aggregator.assistant(),
                history_pruner,
            ]
        )

        max_duration = config.get(
            "max_call_duration_seconds",
            900,
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                audio_in_sample_rate=(
                    16000
                    if call_type == "web"
                    else config["audio"]["sample_rate"]
                ),
                audio_out_sample_rate=(
                    16000
                    if call_type == "web"
                    else config["audio"]["sample_rate"]
                ),
                allow_interruptions=config[
                    "audio"
                ][
                    "enable_interruptions"
                ],
                enable_metrics=config[
                    "metrics"
                ][
                    "enable_metrics"
                ],
                enable_usage_metrics=config[
                    "metrics"
                ][
                    "enable_usage_metrics"
                ],
            ),
        )

        @transport.event_handler(
            "on_client_connected"
        )
        async def on_client_connected(
            transport,
            client,
        ):
            logger.info(
                "[{}] Connected initial_language=en",
                stream_id,
            )

            if (
                campaign_data
                and campaign_data.get(
                    "greeting"
                )
            ):
                customer_name = campaign_data.get(
                    "customer_name",
                    "there",
                )

                greeting = (
                    campaign_data[
                        "greeting"
                    ].replace(
                        "{customer_name}",
                        customer_name,
                    )
                )
            else:
                greeting = config.get(
                    f"greeting_{call_type}",
                    "Say a brief greeting.",
                )

            messages.append(
                {
                    "role": "user",
                    "content": greeting,
                }
            )

            await task.queue_frames(
                [
                    OpenAILLMContextFrame(
                        context
                    )
                ]
            )

            async def hard_timeout():
                await asyncio.sleep(
                    max_duration
                )

                if shutdown_state[
                    "active"
                ]:
                    return

                shutdown_state[
                    "active"
                ] = True

                logger.warning(
                    "[{}] Hard call timeout",
                    stream_id,
                )

                await force_provider_hangup(
                    "hard_timeout"
                )

                await task.cancel()

            asyncio.create_task(
                hard_timeout()
            )

        @transport.event_handler(
            "on_client_disconnected"
        )
        async def on_client_disconnected(
            transport,
            client,
        ):
            snapshot = (
                language_state.snapshot()
            )

            logger.info(
                "[{}] Disconnected language={} "
                "turns={} switches={}",
                stream_id,
                snapshot["current_language"],
                snapshot["turn_index"],
                snapshot["switch_count"],
            )

            await task.cancel()

        runner = PipelineRunner(
            handle_sigint=False
        )

        try:
            await runner.run(task)

        except asyncio.CancelledError:
            raise

        except Exception:
            logger.exception(
                "[{}] Pipeline error",
                stream_id,
            )

    except Exception:
        logger.exception(
            "[{}] Call initialization/runtime failure",
            stream_id,
        )

    finally:
        if not aiohttp_session.closed:
            await aiohttp_session.close()

        logger.remove(
            log_handler
        )