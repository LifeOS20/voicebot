"""
Production voicebot orchestration with:
- Working interruption handling (Pipecat built-in + serializer drop window)
- Low-latency call start (LLM-generated greeting)
- Global VAD reuse for fast cold-start
- Proper Vobiz µ-law <-> PCM conversion
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
import time
from importlib import import_module
from typing import Any, Awaitable, Callable, Optional

import aiohttp
import httpx
from dotenv import load_dotenv
from fastapi import WebSocket
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.frames.frames import (
    AudioRawFrame,
    EndTaskFrame,
    FunctionCallResultProperties,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame, 
    BotStoppedSpeakingFrame,
    LLMContextFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSUpdateSettingsFrame,
    TranscriptionFrame,
    TextFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
)
from pipecat.services.llm_service import FunctionCallParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.turns.user_start import (
    ExternalUserTurnStartStrategy,
)
from pipecat.turns.user_stop import ExternalUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
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
from metrics_collector import CallMetricsCollector

load_dotenv(override=True)
os.makedirs("logs", exist_ok=True)

logger.add(
    "logs/voicebot_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="10 days",
    level="DEBUG",
)

STREAM_PROVIDER_CALL_IDS: dict[str, str] = {}



class InitialOutboundHandshakeProcessor(FrameProcessor):
    """Handle the first caller response without spending an LLM turn on greetings."""

    GREETING_WORDS = {
        "hi", "hello", "hey", "hiya", "namaste", "namaskar",
        "हाय", "हेलो", "नमस्ते", "నమస్తే", "హలో",
    }
    YES_WORDS = {"yes", "yeah", "yep", "yup", "haan", "han", "ji"}

    def __init__(self, *, enabled: bool, customer_name: str | None = None) -> None:
        super().__init__()
        self._enabled = enabled
        self._customer_name = (customer_name or "").strip()
        self._handled_first_transcript = False

    @staticmethod
    def _normalize(text: str) -> list[str]:
        text = " ".join((text or "").strip().lower().split())
        return text.split()

    def _classify(self, text: str) -> str | None:
        tokens = self._normalize(text)
        if not tokens:
            return None

        # Greeting-only responses are the common barge-in case after the
        # application's outbound opening. Do not spend an LLM turn on them.
        if len(tokens) <= 3 and all(token in self.GREETING_WORDS for token in tokens):
            return "greeting"

        # Identity confirmations can also be answered deterministically.
        if len(tokens) <= 4:
            greeting_present = any(token in self.GREETING_WORDS for token in tokens)
            yes_present = any(token in self.YES_WORDS for token in tokens)
            if greeting_present and yes_present:
                return "identity"
            if tokens[0] in self.YES_WORDS:
                return "identity"

        if self._customer_name:
            name_tokens = set(self._normalize(self._customer_name))
            if name_tokens and name_tokens.issubset(set(tokens)) and len(tokens) <= len(name_tokens) + 2:
                return "identity"

        return None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if (
            not self._enabled
            or self._handled_first_transcript
            or direction != FrameDirection.DOWNSTREAM
            or not isinstance(frame, TranscriptionFrame)
        ):
            await self.push_frame(frame, direction)
            return

        text = getattr(frame, "text", "") or ""
        if not text.strip():
            await self.push_frame(frame, direction)
            return

        # Only consume a finalized first transcript. Interim text must continue
        # downstream so normal turn handling remains untouched.
        finalized = getattr(frame, "finalized", True)
        if finalized is False:
            await self.push_frame(frame, direction)
            return

        response_type = self._classify(text)
        self._handled_first_transcript = True

        if response_type == "greeting":
            response = (
                "Hi, yes. This is Ananya from Prestige Group, "
                "calling about Prestige Green Meadows in Whitefield."
            )
        elif response_type == "identity":
            name = f", {self._customer_name}" if self._customer_name else ""
            response = (
                f"Thanks{name}. I’m Ananya from Prestige Group, "
                "calling about Prestige Green Meadows in Whitefield. Do you have a minute?"
            )
        else:
            # Substantive first turns must go through the normal LLM path.
            await self.push_frame(frame, direction)
            return

        logger.debug(
            "Initial outbound handshake handled locally: transcript={!r} response_type={}",
            text,
            response_type,
        )
        await self.push_frame(
            TTSSpeakFrame(text=response, append_to_context=False),
            FrameDirection.DOWNSTREAM,
        )


class LanguageObserver(FrameProcessor):
    """STT-driven language observer without hardcoded wordlists."""

    _LANGUAGE_LABELS = {
        "en": "English",
        "hi": "Hindi",
        "te": "Telugu",
        "ta": "Tamil",
    }

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
            language, probability = self._extract_language(frame)

            if language:
                old_language = self._language_state.current_language
                new_language, switched = self._language_state.observe_stt(
                    language, probability
                )

                if switched:
                    locale = LANGUAGE_LOCALES.get(new_language, "en-IN")
                    label = self._LANGUAGE_LABELS.get(new_language, "English")

                    await self.push_frame(
                        TTSUpdateSettingsFrame(settings={"language": locale}),
                        FrameDirection.DOWNSTREAM,
                    )

                    logger.info(
                        "[{}] Language automatically adapted: old={} new={} ({})",
                        self._stream_id,
                        old_language,
                        label,
                        locale,
                    )

        await self.push_frame(frame, direction)

    @staticmethod
    def _extract_language(
        frame: TranscriptionFrame,
    ) -> tuple[Optional[str], Optional[float]]:
        raw_language = getattr(frame, "language", None)
        raw_probability = getattr(frame, "language_probability", None)

        language = normalize_language(
            getattr(raw_language, "value", raw_language)
        )

        if language:
            return language, _normalize_probability(raw_probability)

        result = getattr(frame, "result", None)
        if isinstance(result, dict):
            data = result.get("data")
            containers = [result, data if isinstance(data, dict) else {}]
            for container in containers:
                raw_language = (
                    container.get("language_code")
                    or container.get("language")
                    or container.get("detected_language")
                )
                raw_probability = (
                    container.get("language_probability")
                    if container.get("language_probability") is not None
                    else container.get("language_confidence")
                )
                if raw_probability is None:
                    raw_probability = container.get("confidence")

                language = normalize_language(raw_language)
                if language:
                    return language, _normalize_probability(raw_probability)

        return None, None


def _normalize_probability(value: object) -> Optional[float]:
    try:
        result = float(value)
        if result > 1:
            result /= 100.0
        return max(0.0, min(1.0, result))
    except (TypeError, ValueError):
        return None


class ServiceFactory:
    """Configures services cleanly and prevents deprecated keyword warnings."""

    @staticmethod
    def _import_class(class_path: str) -> Any:
        module_path, class_name = class_path.rsplit(".", 1)
        return getattr(import_module(module_path), class_name)

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
        registry = config.get("providers", {}).get(service_type, {})
        provider_config = registry.get(provider_name)
        if not provider_config:
            raise ValueError(f"Unknown {service_type} provider: {provider_name}")

        class_path = provider_config.get("class_path")
        if not class_path:
            raise ValueError(f"Missing class_path for {service_type}:{provider_name}")

        service_class = cls._import_class(class_path)
        kwargs: dict[str, Any] = {}

        api_key_env = provider_config.get("api_key_env")
        if api_key_env:
            api_key = os.getenv(api_key_env)
            if not api_key:
                raise ValueError(f"Missing environment variable: {api_key_env}")
            kwargs["api_key"] = api_key

        if provider_config.get("_needs_aiohttp", False):
            if aiohttp_session is None:
                raise RuntimeError(f"{service_type}:{provider_name} requires aiohttp_session")
            kwargs["aiohttp_session"] = aiohttp_session

        params = dict(provider_config.get("params", {}))
        params.pop("tools", None)

        # Sarvam legacy STT provides its own server-side VAD when enabled.
        if service_type == "stt" and provider_name == "sarvam":
            params["vad_signals"] = True

        if service_type == "stt":
            stt_lang = params.get("language")
            if not stt_lang or stt_lang == "unknown":
                params["language"] = "en-IN"

        if "voice_id" in params and "voice" not in params:
            params["voice"] = params.pop("voice_id")

        if "MurfFalcon2TTSService" in class_path:
            kwargs.update(params)
        else:
            settings_cls = getattr(service_class, "Settings", None)
            if settings_cls:
                init_signature = inspect.signature(service_class.__init__)
                top_level_names = {
                    name for name in init_signature.parameters
                    if name not in ("self", "api_key", "aiohttp_session", "settings", "params", "kwargs")
                }
                top_level_params = {
                    key: value for key, value in params.items()
                    if key in top_level_names and key not in ("model", "voice", "voice_id")
                }
                settings_params = {
                    key: value for key, value in params.items()
                    if key not in top_level_params
                }
                kwargs.update(top_level_params)
                if settings_params:
                    if hasattr(settings_cls, "from_mapping"):
                        kwargs["settings"] = settings_cls.from_mapping(settings_params)
                    else:
                        kwargs["settings"] = settings_cls(**settings_params)
            else:
                input_params_cls = getattr(service_class, "InputParams", None)
                if input_params_cls:
                    kwargs["params"] = input_params_cls(**params)
                else:
                    kwargs.update(params)

        kwargs.update(dynamic_kwargs)
        logger.info("Creating {} provider={} class={}", service_type, provider_name, class_path)
        return service_class(**kwargs)


async def warmup_providers(config: dict, aiohttp_session: aiohttp.ClientSession | None = None) -> None:
    """Pre-warm providers and models to ensure Turn 1 is near-zero cold lag."""
    provider_registry = config.get("providers", {})
    class_paths = sorted(
        {
            provider_config["class_path"]
            for providers_of_type in provider_registry.values()
            for provider_config in providers_of_type.values()
            if provider_config.get("class_path")
        }
    )

    for class_path in class_paths:
        try:
            await asyncio.to_thread(ServiceFactory._import_class, class_path)
        except Exception:
            logger.exception("Warmup: failed to import {}", class_path)

    # Pre-warm aiohttp session connections if provided
    if aiohttp_session and not aiohttp_session.closed:
        warmup_urls = [
            "https://api.sarvam.ai/v1",
            "https://api.groq.com/openai/v1",
            "https://in.api.murf.ai/v1/speech/stream",
            "https://api.deepgram.com/v1/listen",
        ]
        for url in warmup_urls:
            try:
                async with aiohttp_session.get(url, timeout=aiohttp.ClientTimeout(total=2.0)) as _:
                    pass
            except Exception:
                pass

    logger.info("Warmup complete: {} provider classes pre-warmed", len(class_paths))


def _prune_history(messages: list, max_messages: int) -> None:
    if len(messages) <= max_messages:
        return
    system_message = messages[0]
    messages[:] = [system_message, *messages[-(max_messages - 1):]]


class _SpokenTextGuard(FrameProcessor):
    """Normalize typography and ensure contractions render cleanly for TTS."""

    @staticmethod
    def _text(frame: Frame) -> str:
        value = getattr(frame, "text", None)
        if isinstance(value, str):
            return value.replace("’", "'").replace("‘", "'")
        return ""

    async def process_frame(
        self,
        frame: Frame,
        direction: FrameDirection,
    ) -> None:
        await super().process_frame(frame, direction)

        if (
            direction == FrameDirection.DOWNSTREAM
            and isinstance(frame, (TextFrame, TTSSpeakFrame))
        ):
            text = self._text(frame)
            if text == "":
                return

            if isinstance(frame, TextFrame):
                frame.text = text
            elif isinstance(frame, TTSSpeakFrame):
                frame.text = text

        await self.push_frame(frame, direction)


class _InterruptionAudioGate(FrameProcessor):
    """ElevenLabs-style multi-context audio gate: drops stale audio from interrupted utterances."""

    def __init__(
        self,
        *,
        stream_id: str,
        on_interruption: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self._stream_id = stream_id
        self._current_gen_id = 0
        self._active_playing_gen_id = 0
        self._is_interrupted = False
        self._dropped_frames = 0
        self._on_interruption = on_interruption

    def next_generation(self) -> int:
        self._current_gen_id += 1
        self._is_interrupted = False
        return self._current_gen_id

    async def process_frame(
        self,
        frame: Frame,
        direction: FrameDirection,
    ) -> None:
        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, InterruptionFrame):
                self._is_interrupted = True
                self._active_playing_gen_id = 0
                if self._on_interruption:
                    self._on_interruption()
                logger.debug("[{}] Barge-in event: invalidated active audio output", self._stream_id)

            elif isinstance(frame, TTSStartedFrame):
                self._active_playing_gen_id = self._current_gen_id
                self._is_interrupted = False
                if self._dropped_frames:
                    logger.debug(
                        "[{}] Playing fresh utterance; purged {} stale audio frames",
                        self._stream_id,
                        self._dropped_frames,
                    )
                    self._dropped_frames = 0

            elif isinstance(frame, TTSStoppedFrame):
                self._active_playing_gen_id = 0

            elif isinstance(frame, AudioRawFrame):
                if self._is_interrupted or (self._active_playing_gen_id != self._current_gen_id):
                    self._dropped_frames += 1
                    return

        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


class _SilenceChecker(FrameProcessor):
    """
    Checks for prolonged user silence and injects a check-in prompt.
    
    - Tracks when user last spoke (via UserStartedSpeakingFrame / TranscriptionFrame)
    - If silence exceeds threshold, queues a check-in TTSSpeakFrame
    - Resets timer when user speaks again
    - Only triggers once per silence period to avoid spam
    """
    
    def __init__(
        self,
        *,
        stream_id: str,
        task: PipelineTask | None = None,
        context_aggregator_user: Any,
        silence_threshold_secs: float = 5.0,
        check_in_message: str = "Are you still there? I'm listening.",
    ) -> None:
        super().__init__()
        self._stream_id = stream_id
        self._task = task
        self._context_aggregator_user = context_aggregator_user
        self._silence_threshold_secs = silence_threshold_secs
        self._check_in_message = check_in_message
        
        self._last_user_speech_time: float | None = None
        self._check_in_triggered = False
        self._check_task: asyncio.Task | None = None
        self._running = False

    def set_task(self, task: PipelineTask) -> None:
        """Set the task reference after pipeline creation."""
        self._task = task

    def start(self) -> None:
        """Start the silence monitoring loop."""
        if self._running:
            return
        self._running = True
        self._last_user_speech_time = time.monotonic()
        self._check_task = asyncio.create_task(self._monitor_silence())

    def stop(self) -> None:
        """Stop the silence monitoring loop."""
        self._running = False
        if self._check_task and not self._check_task.done():
            self._check_task.cancel()
            self._check_task = None

    def on_user_speech(self) -> None:
        """Call when user starts speaking - resets silence timer."""
        self._last_user_speech_time = time.monotonic()
        self._check_in_triggered = False
        logger.debug(f"[{self._stream_id}] SilenceChecker: user speech detected, timer reset")

    async def _monitor_silence(self) -> None:
        """Background task that checks for prolonged silence."""
        while self._running:
            await asyncio.sleep(1.0)  # Check every second
            
            if self._last_user_speech_time is None:
                continue
                
            elapsed = time.monotonic() - self._last_user_speech_time
            
            if elapsed >= self._silence_threshold_secs and not self._check_in_triggered:
                self._check_in_triggered = True
                logger.info(
                    f"[{self._stream_id}] SilenceChecker: {self._silence_threshold_secs}s silence detected, "
                    f"injecting check-in prompt"
                )
                
                # Inject check-in message via LLM
                try:
                    if self._task:
                        self._task.queue_frames([
                            TTSSpeakFrame(text=self._check_in_message, append_to_context=True),
                        ])
                except Exception as e:
                    logger.error(f"[{self._stream_id}] SilenceChecker: failed to queue check-in: {e}")

    async def process_frame(
        self,
        frame: Frame,
        direction: FrameDirection,
    ) -> None:
        await super().process_frame(frame, direction)

        # Detect user speech from upstream frames
        if direction == FrameDirection.UPSTREAM:
            if isinstance(frame, UserStartedSpeakingFrame):
                self.on_user_speech()
            elif isinstance(frame, TranscriptionFrame) and frame.text.strip():
                self.on_user_speech()

        await self.push_frame(frame, direction)


class _TerminationProcessor(FrameProcessor):
    """Detect farewell text without buffering or rewriting normal LLM output."""

    def __init__(
        self,
        *,
        stream_id: str,
        on_hangup: Callable[[], Awaitable[None]],
        force_hangup_fn: Callable[[str], Awaitable[None]],
        grace_seconds: float = 1.5,
        safety_seconds: float = 15.0,
    ) -> None:
        super().__init__()
        self._stream_id = stream_id
        self._on_hangup = on_hangup
        self._force_hangup_fn = force_hangup_fn
        self._grace_seconds = grace_seconds
        self._safety_seconds = safety_seconds
        self._termination_requested = False
        self._provider_hangup_sent = False
        self._waiting_for_bot_stop = False
        self._hangup_task: asyncio.Task | None = None
        self._shutdown_state: dict[str, bool] = {}
        self._end_call_pending: dict[str, bool] = {}

        self.silent_termination_patterns = [
            ("end_call_leak", re.compile(r"\bend[_\s-]?call\b\.?", re.IGNORECASE)),
            ("tool_narration", re.compile(r"\b(call|dial|invok|trigger|activat)\w*\s+(the\s+)?(tool|function|api)\b", re.IGNORECASE)),
            ("explicit_command", re.compile(r"\b(end|clos|terminat|finish|hang)\w*\s+(the\s+)?(call|conversation|session|up|tool)\b", re.IGNORECASE)),
            ("disposition_leak", re.compile(r"\b(site\s+visit|inventory|lead)\s+(is\s+)?(booked|sent|partial)\b", re.IGNORECASE)),
            ("explicit_tag", re.compile(r"\[hangup\]|\[end\]", re.IGNORECASE)),
        ]

        self.spoken_termination_patterns = [
            re.compile(r"\b(goodbye|bye|good day|take care)\b", re.IGNORECASE),
            re.compile(r"\bhave a (nice|great|good) (day|evening|night)\b", re.IGNORECASE),
            re.compile(r"\btalk to you (later|soon)\b", re.IGNORECASE),
            re.compile(r"\bthank you for your time\b", re.IGNORECASE),
            re.compile(r"\bthanks for your time\b", re.IGNORECASE),
        ]

    def bind_state(
        self,
        *,
        shutdown_state: dict[str, bool],
        end_call_pending: dict[str, bool],
    ) -> None:
        self._shutdown_state = shutdown_state
        self._end_call_pending = end_call_pending

    async def _hangup_after_bot_stop(self) -> None:
        try:
            await asyncio.sleep(self._grace_seconds)
            if not self._provider_hangup_sent:
                await self._force_hangup_fn("farewell_complete")
                self._provider_hangup_sent = True
            await self.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)
        except asyncio.CancelledError:
            return

    async def _schedule_hangup(self, delay: float) -> None:
        try:
            await asyncio.sleep(delay)
            if not self._provider_hangup_sent:
                await self._force_hangup_fn("termination_immediate")
                self._provider_hangup_sent = True
            await self.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)
        except asyncio.CancelledError:
            return

    async def process_frame(
        self,
        frame: Frame,
        direction: FrameDirection,
    ) -> None:
        # A barge-in cancels a pending farewell hangup.
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, InterruptionFrame):
            self._waiting_for_bot_stop = False
            if self._hangup_task and not self._hangup_task.done():
                self._hangup_task.cancel()
            self._hangup_task = None
            self._termination_requested = False
            self._provider_hangup_sent = False
            self._shutdown_state["active"] = False
            self._end_call_pending["active"] = False
            await super().process_frame(frame, direction)
            await self.push_frame(frame, direction)
            return

        # Inspect speech for farewell intent, but pass the original frame through unchanged.
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, (TextFrame, TTSSpeakFrame)):
            text = getattr(frame, "text", "") or ""
            silent_match = any(pattern.search(text) for _, pattern in self.silent_termination_patterns)
            spoken_match = any(pattern.search(text) for pattern in self.spoken_termination_patterns)

            if silent_match:
                self._termination_requested = True
                self._shutdown_state["active"] = True
                if self._hangup_task and not self._hangup_task.done():
                    self._hangup_task.cancel()
                self._hangup_task = asyncio.create_task(self._schedule_hangup(0.5))
                logger.info("[{}] Silent termination detected; forcing hangup", self._stream_id)
            elif spoken_match:
                self._termination_requested = True
                self._waiting_for_bot_stop = True
                self._shutdown_state["active"] = True
                logger.info("[{}] Farewell detected; will hang up after bot stops speaking", self._stream_id)

            await super().process_frame(frame, direction)
            await self.push_frame(frame, direction)
            return

        # Once the farewell has actually finished playing, hang up.
        if direction == FrameDirection.UPSTREAM and isinstance(frame, BotStoppedSpeakingFrame):
            if self._waiting_for_bot_stop and not self._hangup_task:
                logger.info("[{}] Farewell finished; hanging up in {}s", self._stream_id, self._grace_seconds)
                self._hangup_task = asyncio.create_task(self._hangup_after_bot_stop())
                self._waiting_for_bot_stop = False

            await super().process_frame(frame, direction)
            await self.push_frame(frame, direction)
            return

        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


class _CallEndCoordinator(FrameProcessor):
    """Graceful call termination handler with grace window support."""

    def __init__(
        self,
        *,
        stream_id: str,
        on_hangup: Callable[[], Awaitable[None]],
        grace_seconds: float = 1.4,
        safety_seconds: float = 20.0,
    ) -> None:
        super().__init__()
        self._stream_id = stream_id
        self._on_hangup = on_hangup
        self._grace_seconds = grace_seconds
        self._safety_seconds = safety_seconds

        self._requested = False
        self._awaiting_closing = False
        self._closing_in_progress = False
        self._in_grace = False
        self._ended = False

        self._grace_task: asyncio.Task | None = None
        self._safety_task: asyncio.Task | None = None

    @property
    def is_ending(self) -> bool:
        return self._requested

    def request_ending(self) -> None:
        self._requested = True
        self._awaiting_closing = True
        self._closing_in_progress = False
        self._in_grace = False
        self._cancel_task("_safety_task")
        self._safety_task = asyncio.create_task(self._run_safety_timeout())
        logger.info("[{}] Call ending requested; waiting for closing line", self._stream_id)

    def cancel_ending(self, reason: str) -> None:
        if not self._requested:
            return
        logger.info("[{}] Call ending cancelled ({})", self._stream_id, reason)
        self._requested = False
        self._awaiting_closing = False
        self._closing_in_progress = False
        self._in_grace = False
        self._cancel_task("_grace_task")
        self._cancel_task("_safety_task")

    def _cancel_task(self, attr: str) -> None:
        task = getattr(self, attr)
        if task is not None and not task.done():
            task.cancel()
        setattr(self, attr, None)

    async def _run_safety_timeout(self) -> None:
        try:
            await asyncio.sleep(self._safety_seconds)
        except asyncio.CancelledError:
            return
        logger.warning(
            "[{}] Closing statement timed out ({}s); forcing hangup",
            self._stream_id,
            self._safety_seconds,
        )
        await self._finish()

    async def _run_grace_timeout(self) -> None:
        try:
            await asyncio.sleep(self._grace_seconds)
        except asyncio.CancelledError:
            return
        await self._finish()

    async def _finish(self) -> None:
        if self._ended:
            return
        self._ended = True
        self._cancel_task("_grace_task")
        self._cancel_task("_safety_task")
        await self._on_hangup()

    async def process_frame(
        self,
        frame: Frame,
        direction: FrameDirection,
    ) -> None:
        await super().process_frame(frame, direction)

        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, InterruptionFrame) and (
                self._awaiting_closing or self._closing_in_progress or self._in_grace
            ):
                self.cancel_ending("caller spoke during closing")

            elif isinstance(frame, TTSStartedFrame) and self._awaiting_closing:
                self._awaiting_closing = False
                self._closing_in_progress = True
                logger.debug("[{}] Closing statement started speaking", self._stream_id)

            elif isinstance(frame, TTSStoppedFrame) and self._closing_in_progress:
                self._closing_in_progress = False
                self._in_grace = True
                logger.debug(
                    "[{}] Closing statement finished; {}s grace window open",
                    self._stream_id,
                    self._grace_seconds,
                )
                self._cancel_task("_grace_task")
                self._grace_task = asyncio.create_task(self._run_grace_timeout())

        await self.push_frame(frame, direction)


class _HistoryPruner(FrameProcessor):
    """Keeps the LLM context bounded across long calls."""

    def __init__(
        self,
        context: LLMContext,
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


async def _create_provider_services_in_parallel(
    config: dict,
    active: dict,
    *,
    sample_rate: int,
    aiohttp_session: aiohttp.ClientSession,
) -> tuple[Any, Any, Any]:
    stt_task = asyncio.to_thread(
        ServiceFactory.create,
        "stt",
        active["stt"],
        config,
        sample_rate=sample_rate,
    )
    llm_task = asyncio.to_thread(
        ServiceFactory.create,
        "llm",
        active["llm"],
        config,
    )
    tts_task = asyncio.to_thread(
        ServiceFactory.create,
        "tts",
        active["tts"],
        config,
        sample_rate=sample_rate,
        aiohttp_session=aiohttp_session,
    )

    stt, llm, tts = await asyncio.gather(
        stt_task,
        llm_task,
        tts_task,
    )

    return stt, llm, tts


async def run_bot(
    websocket: WebSocket,
    call_type: str,
    config: dict,
    stream_id: str | None = None,
    campaign_data: dict | None = None,
    call_id: str | None = None,
):
    provider_call_id = (
        campaign_data.get("provider_call_id") if campaign_data else None
    )

    aiohttp_session = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=60, connect=10),
        connector=aiohttp.TCPConnector(limit=20, ttl_dns_cache=300),
    )

    # Pre-warm providers and connection pools in parallel with handshake
    warmup_task = asyncio.create_task(warmup_providers(config, aiohttp_session))

    active = config["active_providers"]
    sample_rate = 16000 if call_type == "web" else config["audio"]["sample_rate"]

    provider_construction_task = asyncio.create_task(
        _create_provider_services_in_parallel(
            config,
            active,
            sample_rate=sample_rate,
            aiohttp_session=aiohttp_session,
        )
    )

    # Vobiz Handshake - read initial messages to get stream_id
    if not stream_id:
        try:
            for _ in range(5):
                message = await websocket.receive_text()
                data = json.loads(message)

                candidate_stream_id = data.get("streamId") or data.get("start", {}).get("streamId")
                candidate_call_id = data.get("callId") or data.get("start", {}).get("callId")

                if candidate_call_id:
                    provider_call_id = candidate_call_id

                if candidate_stream_id:
                    stream_id = candidate_stream_id
                    break

            if not stream_id:
                stream_id = "unknown"

        except Exception:
            logger.exception("Failed to parse initial Vobiz messages")
            stream_id = "unknown_error"

    call_label = call_id or stream_id or "no-call-id"

    if stream_id and provider_call_id:
        STREAM_PROVIDER_CALL_IDS[stream_id] = provider_call_id
    elif stream_id:
        provider_call_id = STREAM_PROVIDER_CALL_IDS.get(stream_id)

    conversation_call_type = "outbound" if call_type == "web" else call_type

    logger.info(
        "[{}] Starting transport={} conversation_type={} call_id={} providers={}",
        stream_id,
        call_type,
        conversation_call_type,
        call_label,
        config.get("active_providers", {}),
    )

    async def force_provider_hangup(trigger: str) -> None:
        if not provider_call_id:
            logger.warning("[{}] Missing Vobiz call ID ({})", stream_id, trigger)
            return

        auth_id = os.getenv("VOBIZ_AUTH_ID")
        auth_token = os.getenv("VOBIZ_AUTH_TOKEN")

        if not auth_id or not auth_token:
            logger.warning("[{}] Missing Vobiz auth credentials ({})", stream_id, trigger)
            return

        url = f"https://api.vobiz.ai/api/v1/Account/{auth_id}/Call/{provider_call_id}/"

        # Retry logic: up to 3 attempts with exponential backoff
        max_retries = 3
        base_delay = 2.0  # seconds

        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.delete(
                        url,
                        headers={
                            "X-Auth-ID": auth_id,
                            "X-Auth-Token": auth_token,
                            "Content-Type": "application/json",
                        },
                    )
                if response.status_code in {200, 201, 202, 204}:
                    logger.info("[{}] Vobiz hangup succeeded on attempt {} ({})", stream_id, attempt + 1, trigger)
                    return
                else:
                    logger.warning(
                        "[{}] Vobiz hangup failed status={} on attempt {} ({})",
                        stream_id,
                        response.status_code,
                        attempt + 1,
                        trigger,
                    )
            except Exception as e:
                logger.warning(
                    "[{}] Vobiz hangup request error on attempt {} ({}): {}",
                    stream_id,
                    attempt + 1,
                    trigger,
                    e,
                )

            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.info("[{}] Retrying Vobiz hangup in {}s...", stream_id, delay)
                await asyncio.sleep(delay)

        logger.error(
            "[{}] Vobiz hangup FAILED after {} attempts ({}). PSTN call may still be connected and billing!",
            stream_id,
            max_retries,
            trigger,
        )

    log_handler = logger.add(
        f"logs/call_{call_label}_{stream_id}.log",
        level="DEBUG",
    )

    call_metrics = None

    try:
        serializer = (
            WebPCMFrameSerializer(stream_id=stream_id)
            if call_type == "web"
            else VobizFrameSerializer(stream_id=stream_id, sample_rate=8000)
        )

        transport = FastAPIWebsocketTransport(
            websocket=websocket,
            params=FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                add_wav_header=False,
                vad_enabled=False,
                vad_audio_passthrough=(active.get("stt") == "deepgram"),
                serializer=serializer,
            ),
        )

        # Wait for both provider construction and warmup to complete
        stt, llm, tts = await provider_construction_task
        await warmup_task  # Ensure connection pools are primed

        call_metrics = CallMetricsCollector(
            call_id=stream_id,
            stt_provider=active["stt"],
            llm_provider=active["llm"],
            tts_provider=active["tts"],
        )

        language_state = LanguageState(
            sustained_switch_turns=config.get("language", {}).get(
                "sustained_switch_turns", 2
            ),
        )

        shutdown_state = {"active": False}
        end_call_pending = {"active": False}

        async def _perform_end_of_call_hangup() -> None:
            if shutdown_state["active"]:
                return
            shutdown_state["active"] = True
            logger.info("[{}] Ending call now (closing statement completed)", stream_id)
            await force_provider_hangup("call_end_coordinator")
            await task.cancel()

        termination_processor = _TerminationProcessor(
            stream_id=stream_id,
            on_hangup=_perform_end_of_call_hangup,
            force_hangup_fn=force_provider_hangup,
            grace_seconds=1.5,
            safety_seconds=15.0,
        )
        # Inject references for coordination with end_call handler
        termination_processor.bind_state(
            shutdown_state=shutdown_state,
            end_call_pending=end_call_pending,
        )

        # FIX: call_end_coordinator was referenced inside end_call() but never
        # instantiated, which caused "name 'call_end_coordinator' is not defined"
        # on every attempted hangup.
        call_end_coordinator = _CallEndCoordinator(
            stream_id=stream_id,
            on_hangup=_perform_end_of_call_hangup,
            grace_seconds=1.5,
            safety_seconds=15.0,
        )

        interruption_audio_gate = _InterruptionAudioGate(
            stream_id=stream_id,
            on_interruption=serializer.on_interruption if hasattr(serializer, "on_interruption") else None,
        )

        def _latest_user_utterance() -> str:
            for message in reversed(messages):
                if message.get("role") == "user":
                    content = message.get("content", "")
                    return content.strip() if isinstance(content, str) else ""
            return ""

        async def end_call(params: FunctionCallParams):
            tool_call_started_at = time.monotonic()
            latest_user_text = _latest_user_utterance()

            if not latest_user_text:
                logger.warning("[{}] Rejected end_call before first caller utterance", stream_id)
                await params.result_callback(
                    {"status": "rejected", "reason": "no caller utterance yet"},
                    properties=FunctionCallResultProperties(run_llm=True),
                )
                return

            if call_end_coordinator.is_ending:
                logger.info("[{}] Ignored duplicate end_call while already ending", stream_id)
                await params.result_callback(
                    {"status": "already_ending"},
                    properties=FunctionCallResultProperties(run_llm=False),
                )
                return

            logger.info(
                "[{}] end_call requested language={} latest_user_utterance={!r}",
                stream_id,
                language_state.current_language,
                latest_user_text,
            )

            call_end_coordinator.request_ending()

            try:
                await params.result_callback(
                    {
                        "status": "ending",
                        "say": (
                            "Speak one brief, warm closing line now that fits what was "
                            "actually discussed on this call, thank the caller, and do not call any tool this turn."
                        ),
                    },
                    properties=FunctionCallResultProperties(run_llm=False),
                )
            except Exception:
                call_end_coordinator.cancel_ending("end_call setup raised an exception")
                call_metrics.record_tool_call(
                    "end_call",
                    success=False,
                    latency_ms=(time.monotonic() - tool_call_started_at) * 1000,
                )
                raise
            else:
                call_metrics.record_tool_call(
                    "end_call",
                    success=True,
                    latency_ms=(time.monotonic() - tool_call_started_at) * 1000,
                )

        llm.register_function("end_call", end_call, cancel_on_interruption=True)

        # Build concise campaign prompt
        system_prompt, customer_context = build_system_prompt(
            conversation_call_type,
            config,
            campaign_data,
        )

        messages = [{"role": "system", "content": system_prompt}]
        if customer_context:
            messages.append({"role": "system", "content": customer_context})

        llm_tools_raw = (
            config.get("providers", {})
            .get("llm", {})
            .get(active["llm"], {})
            .get("params", {})
            .get("tools", [])
        )
        llm_tools_raw = [
            raw for raw in llm_tools_raw
            if not (isinstance(raw, dict) and raw.get("function", {}).get("name") == "set_conversation_language")
        ]

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
        turn_config = config.get("turn_management", {})

        # Turn aggregation with faster endpointing
        context_aggregator = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(
                user_turn_strategies=UserTurnStrategies(
                    start=[
                        ExternalUserTurnStartStrategy(),
                    ],
                    stop=[
                        ExternalUserTurnStopStrategy(
                            timeout=0.2,
                            wait_for_transcript=True,
                        ),
                    ],
                ),
                user_turn_stop_timeout=float(
                    turn_config.get("user_turn_stop_timeout", 1.0)
                ),
            ),
        )

        language_observer = LanguageObserver(
            stream_id=stream_id,
            language_state=language_state,
            tts=tts,
            tts_provider=active["tts"],
        )

        history_pruner = _HistoryPruner(
            context=context,
            max_messages=config.get("max_conversation_messages", 20),
            stream_id=stream_id,
        )

        spoken_text_guard = _SpokenTextGuard()

        customer_name_for_handshake = None
        if campaign_data:
            customer_name_for_handshake = campaign_data.get("customer_name")
        if not customer_name_for_handshake and call_type == "web":
            customer_name_for_handshake = config.get("test_customer_name")

        initial_outbound_handshake = InitialOutboundHandshakeProcessor(
            enabled=(conversation_call_type == "outbound"),
            customer_name=customer_name_for_handshake,
        )

        # Pipeline components - include silence_checker (will set task later)
        silence_checker = _SilenceChecker(
            stream_id=stream_id,
            task=None,  # Will set after task creation
            context_aggregator_user=context_aggregator.user(),
            silence_threshold_secs=5.0,
            check_in_message="Are you still there?",
        )

        pipeline = Pipeline([
            transport.input(),
            stt,
            language_observer,
            initial_outbound_handshake,
            context_aggregator.user(),
            llm,
            spoken_text_guard,
            tts,
            interruption_audio_gate,
            call_end_coordinator,
            termination_processor,
            silence_checker,
            transport.output(),
            context_aggregator.assistant(),
            history_pruner,
            call_metrics,
        ])

        max_duration = config.get("max_call_duration_seconds", 900)

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                audio_in_sample_rate=(16000 if call_type == "web" else config["audio"]["sample_rate"]),
                audio_out_sample_rate=(16000 if call_type == "web" else config["audio"]["sample_rate"]),
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
        )

        # Set task reference on silence_checker now that task exists
        silence_checker.set_task(task)

        # Track whether we've sent the initial greeting
        _greeting_sent = {"done": False}

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            if _greeting_sent["done"]:
                logger.debug("[{}] Greeting already sent, skipping", stream_id)
                return

            logger.info("[{}] Client connected, sending greeting via LLM", stream_id)

            customer_name = None
            if campaign_data:
                customer_name = campaign_data.get("customer_name")
            if not customer_name and call_type == "web":
                customer_name = config.get("test_customer_name")

            if conversation_call_type == "outbound":
                if customer_name:
                    greeting_template = config.get(
                        "greeting_outbound",
                        "Hi, am I speaking with {customer_name}?",
                    )
                    greeting = str(greeting_template).replace("{customer_name}", customer_name)
                else:
                    greeting = str(
                        config.get(
                            "greeting_outbound_no_name",
                            "Hi, is this a good time to talk?",
                        )
                    )
            else:
                greeting = str(
                    config.get(
                        f"greeting_{conversation_call_type}",
                        config.get("greeting_inbound", "Hi, how can I help you?"),
                    )
                )

            # OLD WORKING APPROACH: Add greeting as user message, trigger LLM to respond
            # This lets the LLM generate the greeting naturally with proper context
            # Personalized greeting is already built above
            greeting = (
                f"Hi, am I speaking with {customer_name}?"
            )
            messages.append({"role": "assistant", "content": greeting})

            interruption_audio_gate.next_generation()
            if hasattr(serializer, "next_generation"):
                serializer.next_generation()

            _greeting_sent["done"] = True

            await task.queue_frames([
                TTSSpeakFrame(text=greeting, append_to_context=False),
            ])

            async def hard_timeout():
                await asyncio.sleep(max_duration)
                if shutdown_state["active"]:
                    return
                shutdown_state["active"] = True
                logger.warning("[{}] Hard call timeout", stream_id)
                await force_provider_hangup("hard_timeout")
                await task.cancel()

            asyncio.create_task(hard_timeout())

        # Bump generation counter every time a user turn concludes
        @context_aggregator.user().event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(processor, strategy, *args, **kwargs):
            gen_id = interruption_audio_gate.next_generation()
            if hasattr(serializer, "next_generation"):
                serializer.next_generation()
            logger.debug("[{}] Advanced audio generation context to id={}", stream_id, gen_id)

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            snapshot = language_state.snapshot()
            logger.info(
                "[{}] Disconnected language={} turns={} switches={}",
                stream_id,
                snapshot["current_language"],
                snapshot["turn_index"],
                snapshot["switch_count"],
            )
            silence_checker.stop()
            await task.cancel()

        runner = PipelineRunner(handle_sigint=False)

        try:
            await runner.run(task)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[{}] Pipeline error", stream_id)

    except Exception:
        logger.exception("[{}] Call initialization/runtime failure", stream_id)

    finally:
        STREAM_PROVIDER_CALL_IDS.pop(stream_id, None)

        if call_metrics is not None:
            call_metrics.finalize()

        if not aiohttp_session.closed:
            await aiohttp_session.close()

        logger.remove(log_handler)