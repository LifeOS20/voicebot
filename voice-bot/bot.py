"""
Production voicebot orchestration with:
- Working interruption handling (Pipecat built-in + serializer drop window)
- Low-latency deterministic outbound call start
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
from openai import AsyncOpenAI

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.frames.frames import (
    AudioRawFrame,
    EndTaskFrame,
    FunctionCallResultProperties,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    BotStartedSpeakingFrame,
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


# ---------------------------------------------------------------------------
# NEW: deterministic backstop for premature end_call.
#
# config.yaml's termination_rules already tells the model (in prose) not to
# end the call on agreement words, budget mentions, or stated configuration/
# purpose preferences. The Sept 1 investment-call log proves that a prose
# rule is not enough: qwen/qwen3.8-27b runs with reasoning_effort=none and
# temperature=0.7, so the tool-call decision is a single, sampled forward
# pass with no chain-of-thought -- it can and did call end_call() with zero
# spoken text the instant the caller said "invest karna tha" (a *positive*
# expression of interest that the prompt explicitly calls out as a non-end
# case). A stochastic 27B model cannot be trusted to enforce a business-
# critical binary decision through prompting alone, so we enforce it here
# in code instead, as a hard veto in front of call_end_coordinator.
# request_ending(). This does not replace termination_rules in the prompt;
# it backstops it.
# ---------------------------------------------------------------------------
_END_CALL_EXPLICIT_PATTERNS = [
    re.compile(
        r"\b("
        r"bye|goodbye|good\s+bye|see\s+you|take\s+care|"
        r"not\s+interested|no\s+thanks|"
        r"please\s+stop\s+calling|stop\s+calling|"
        r"don't\s+call\s+me|do\s+not\s+call\s+me|"
        r"i\s+don't\s+want\s+to\s+continue|"
        r"i\s+do\s+not\s+want\s+to\s+continue|"
        r"that's\s+all|that\s+is\s+all"
        r")\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b("
        r"nahi\s+chahiye|"
        r"mujhe\s+interest\s+nahi\s+hai|"
        r"interest\s+nahi\s+hai|"
        r"baat\s+nahi\s+karni|"
        r"call\s+mat\s+karna|"
        r"dobara\s+call\s+mat\s+karna|"
        r"bas\s+itna\s+hi|"
        r"bas\s+ho\s+gaya"
        r")\b",
        re.IGNORECASE,
    ),
]


def _caller_explicitly_ended(text: str) -> bool:
    """Return True only when the caller clearly ends or declines the call."""
    text = (text or "").strip()
    if not text:
        return False
    return any(
        pattern.search(text)
        for pattern in _END_CALL_EXPLICIT_PATTERNS
    )


# ---------------------------------------------------------------------------
# NEW: strip a trailing dynamic "Current time: ..." suffix from the system
# prompt before it's used anywhere, including in the cache pre-warm call.
#
# Groq's prompt cache requires an exact, character-for-character prefix
# match across requests (see console.groq.com/docs/prompt-caching). Every
# call currently gets a system prompt with a per-call wall-clock timestamp
# baked into the end of it (visible in the logs: "...campaign context.\n\n
# Current time: 2026-09-01 11:00:56"). That timestamp changes every call
# and even within a call's own pre-warm-vs-real-turn pair, which means the
# "stable" prefix was never actually stable -- every single request was
# guaranteed a cache miss on the tail of the prompt regardless of whether
# caching is enabled for this model at all. This is a bot.py-level strip
# rather than a prompt_builder.py edit, since prompt_builder.py wasn't
# provided; if the current-time line is ever load-bearing for the model's
# behavior, this should instead be moved to a separate, explicitly dynamic
# system message appended after the stable prefix (see prompt_caching
# structure note in the module docstring below), not embedded inline.
# ---------------------------------------------------------------------------
_TRAILING_CURRENT_TIME_RE = re.compile(r"\n\nCurrent time: .*$")


def _strip_dynamic_time_suffix(prompt: str) -> str:
    return _TRAILING_CURRENT_TIME_RE.sub("", prompt)


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
            raise ValueError(
                f"Unknown {service_type} provider: {provider_name}"
            )

        class_path = provider_config.get("class_path")

        if not class_path:
            raise ValueError(
                f"Missing class_path for {service_type}:{provider_name}"
            )

        service_class = cls._import_class(class_path)
        kwargs: dict[str, Any] = {}

        api_key_env = provider_config.get("api_key_env")

        if api_key_env:
            api_key = os.getenv(api_key_env)

            if not api_key:
                raise ValueError(
                    f"Missing environment variable: {api_key_env}"
                )

            kwargs["api_key"] = api_key

        if provider_config.get("_needs_aiohttp", False):
            if aiohttp_session is None:
                raise RuntimeError(
                    f"{service_type}:{provider_name} requires aiohttp_session"
                )

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
                init_signature = inspect.signature(
                    service_class.__init__
                )

                top_level_names = {
                    name
                    for name in init_signature.parameters
                    if name not in (
                        "self",
                        "api_key",
                        "aiohttp_session",
                        "settings",
                        "params",
                        "kwargs",
                    )
                }

                top_level_params = {
                    key: value
                    for key, value in params.items()
                    if key in top_level_names
                    and key not in ("model", "voice", "voice_id")
                }

                settings_params = {
                    key: value
                    for key, value in params.items()
                    if key not in top_level_params
                }

                kwargs.update(top_level_params)

                if settings_params:
                    if hasattr(settings_cls, "from_mapping"):
                        kwargs["settings"] = settings_cls.from_mapping(
                            settings_params
                        )
                    else:
                        kwargs["settings"] = settings_cls(
                            **settings_params
                        )

            else:
                input_params_cls = getattr(
                    service_class,
                    "InputParams",
                    None,
                )

                if input_params_cls:
                    kwargs["params"] = input_params_cls(**params)
                else:
                    kwargs.update(params)

        kwargs.update(dynamic_kwargs)

        logger.info(
            "Creating {} provider={} class={}",
            service_type,
            provider_name,
            class_path,
        )

        return service_class(**kwargs)


async def warmup_providers(
    config: dict,
    aiohttp_session: aiohttp.ClientSession | None = None,
) -> None:
    """
    Pre-warm providers and connection pools to reduce cold-start lag.

    CHANGED: previously imported and pinged all 10 registered provider
    classes across every service type (deepgram, openai, cerebras,
    elevenlabs, murf, etc.) regardless of which 3 are actually active for
    this call (sarvam STT, groq LLM, sarvam TTS per active_providers in
    config.yaml). Every unused provider warmed here was pure wasted time
    sitting in front of the outbound greeting. Now scoped to only the
    active providers.

    CHANGED: the health-check ping loop was sequential
    (`for url in warmup_urls: await session.get(...)`), which is up to
    4 * 2s = 8s worst case, serially, before this function could return.
    Now fired concurrently with asyncio.gather.
    """

    provider_registry = config.get("providers", {})
    active_providers = config.get("active_providers", {})

    class_paths = sorted(
        {
            provider_registry.get(service_type, {})
            .get(provider_name, {})
            .get("class_path")
            for service_type, provider_name in active_providers.items()
        }
        - {None}
    )

    for class_path in class_paths:
        try:
            await asyncio.to_thread(
                ServiceFactory._import_class,
                class_path,
            )

        except Exception:
            logger.exception(
                "Warmup: failed to import {}",
                class_path,
            )

    if aiohttp_session and not aiohttp_session.closed:
        warmup_urls = [
            "https://api.sarvam.ai/v1",
            "https://api.groq.com/openai/v1",
            "https://in.api.murf.ai/v1/speech/stream",
            "https://api.deepgram.com/v1/listen",
        ]

        async def _ping(url: str) -> None:
            try:
                async with aiohttp_session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=2.0),
                ):
                    pass

            except Exception:
                pass

        await asyncio.gather(*(_ping(url) for url in warmup_urls))

    logger.info(
        "Warmup complete: {} provider classes pre-warmed",
        len(class_paths),
    )


def _prune_history(messages: list, max_messages: int) -> None:
    if len(messages) <= max_messages:
        return

    system_message = messages[0]

    messages[:] = [
        system_message,
        *messages[-(max_messages - 1):],
    ]


class _SpokenTextGuard(FrameProcessor):
    """
    Normalizes text right before it reaches TTS so Sarvam speaks natural
    language instead of literal symbols or single characters.

    FIXED: this class used to contain a full copy of _TerminationProcessor's
    process_frame body (interruption handling, farewell regex matching,
    hangup scheduling) but never defined __init__, so every attribute it
    touched (self.silent_termination_patterns, self._waiting_for_bot_stop,
    self._hangup_task, ...) never existed on the instance. Every frame that
    passed through it -- including the very first outbound greeting --
    raised AttributeError, which is exactly the crash in the Sept 1 log
    ('_SpokenTextGuard' object has no attribute 'silent_termination_
    patterns'). Termination detection already runs correctly downstream
    in _TerminationProcessor; it never belonged here, and duplicating it
    here only broke things.

    This class now does what its docstring always claimed: normalize text
    for speech. That's the direct fix for two production symptoms:
      - Literal symbols read aloud ("greater than", "less than", "and")
        when the LLM emits text like "18 < income < 25000" or "EMI & SIP".
      - Fragmented, letter-by-letter speech: with every frame erroring out
        above, TTS was getting an inconsistent, gappy stream of text
        instead of clean, complete phrases. Fixing the crash restores a
        normal flow of complete text into TTS's own chunking/buffering.
    """

    _REPLACEMENTS: list[tuple[re.Pattern, str]] = [
        (re.compile(r"\s*>=\s*"), " at least "),
        (re.compile(r"\s*<=\s*"), " at most "),
        (re.compile(r"\s*>\s*"), " greater than "),
        (re.compile(r"\s*<\s*"), " less than "),
        (re.compile(r"\s*&\s*"), " and "),
        (re.compile(r"\s*%\s*"), " percent "),
        (re.compile(r"[*_`#]+"), ""),          # markdown emphasis/headers
        (re.compile(r"[~^|]"), ""),
        (re.compile(r"\s{2,}"), " "),
    ]

    @classmethod
    def _normalize(cls, text: str) -> str:
        text = (
            text
            .replace("’", "'")
            .replace("‘", "'")
            .replace("“", '"')
            .replace("”", '"')
            .replace("—", ", ")
            .replace("–", " to ")
        )

        for pattern, replacement in cls._REPLACEMENTS:
            text = pattern.sub(replacement, text)

        # FIXED: this used to `return text.strip()`. The LLM streams its
        # reply as a sequence of separate TextFrame deltas ("Hi", " Simran,",
        # " this", " is", " Ananya", ...), and TTS's own downstream
        # aggregator reassembles the full sentence by concatenating those
        # deltas as-is. Stripping each delta's leading/trailing whitespace
        # here deleted the space *between* every pair of chunks before they
        # were ever glued back together -- producing exactly the run-on
        # "HiSimran,thisisAnanyafromPrestigeGroup..." string that got sent
        # to TTS and very likely garbled "Ananya" into "Anya" once the
        # phonetic model had no word boundary left to find it with. Only
        # the substitutions above should touch the text; the boundaries
        # are not this processor's to remove.
        return text

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
            original = getattr(frame, "text", None)

            if isinstance(original, str) and original:
                frame.text = self._normalize(original)

        await self.push_frame(frame, direction)


class _InterruptionAudioGate(FrameProcessor):
    """Drops stale audio from interrupted utterances."""

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

        # FIXED: _active_playing_gen_id used to only get set when a
        # TTSStartedFrame for the new generation arrived at this
        # processor. Between the moment next_generation() bumped
        # _current_gen_id and that TTSStartedFrame showing up, any real
        # AudioRawFrame for the NEW generation that arrived first was
        # compared against the OLD _active_playing_gen_id, failed the
        # match, and got dropped as "stale" -- even though it was the
        # brand-new utterance's own audio. This is provable from
        # production logs: "Playing fresh utterance; purged 10 stale
        # audio frames" fired 4ms after the very first TTS call of a
        # call started generating, before any previous utterance existed
        # to be stale from. On a cold pipeline this window is at its
        # widest on the very first utterance of every call -- the
        # outbound greeting. Setting _active_playing_gen_id here, not
        # just on TTSStartedFrame, closes that window: a fresh generation
        # is presumed current the moment it's declared, and the
        # TTSStartedFrame handler below just reconfirms it.
        self._active_playing_gen_id = self._current_gen_id

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

                logger.debug(
                    "[{}] Barge-in event: invalidated active audio output",
                    self._stream_id,
                )

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
                if (
                    self._is_interrupted
                    or (
                        self._active_playing_gen_id
                        != self._current_gen_id
                    )
                ):
                    self._dropped_frames += 1
                    return

        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


class _SilenceChecker(FrameProcessor):
    """
    Checks for prolonged user silence and injects a check-in prompt.

    - Tracks when user last spoke.
    - If silence exceeds threshold, queues a check-in TTSSpeakFrame.
    - Resets timer when user speaks again.
    - Only triggers once per silence period.
    """

    def __init__(
        self,
        *,
        stream_id: str,
        task: PipelineTask | None = None,
        context_aggregator_user: Any,
        silence_threshold_secs: float = 5.0,
        check_in_message: str = "Are you still there?",
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
        # NEW: tracks whether the bot is currently mid-utterance. Without
        # this, a long bot monologue (8-11s turns are routine in the Sept 1
        # logs) can burn past silence_threshold_secs on its own and fire
        # "Are you still there?" directly on top of the bot's own audio.
        self._bot_is_speaking = False

    def set_task(self, task: PipelineTask) -> None:
        """Set the task reference after pipeline creation."""
        self._task = task

    def start(self) -> None:
        """Start the silence monitoring loop."""
        if self._running:
            return

        self._running = True
        self._last_user_speech_time = time.monotonic()

        self._check_task = asyncio.create_task(
            self._monitor_silence()
        )

    def stop(self) -> None:
        """Stop the silence monitoring loop."""
        self._running = False

        if (
            self._check_task
            and not self._check_task.done()
        ):
            self._check_task.cancel()
            self._check_task = None

    def on_user_speech(self) -> None:
        """Reset silence timer when user speaks."""
        self._last_user_speech_time = time.monotonic()
        self._check_in_triggered = False

        logger.debug(
            "[{}] SilenceChecker: user speech detected, timer reset",
            self._stream_id,
        )

    async def _monitor_silence(self) -> None:
        """Background task that checks for prolonged silence."""

        while self._running:
            await asyncio.sleep(1.0)

            if self._last_user_speech_time is None:
                continue

            elapsed = (
                time.monotonic()
                - self._last_user_speech_time
            )

            if (
                elapsed >= self._silence_threshold_secs
                and not self._check_in_triggered
                # NEW: never check in while the bot itself is speaking --
                # see _bot_is_speaking comment in __init__.
                and not self._bot_is_speaking
            ):
                self._check_in_triggered = True

                logger.info(
                    "[{}] SilenceChecker: {}s silence detected, injecting check-in prompt",
                    self._stream_id,
                    self._silence_threshold_secs,
                )

                try:
                    if self._task:
                        self._task.queue_frames(
                            [
                                TTSSpeakFrame(
                                    text=self._check_in_message,
                                    append_to_context=True,
                                ),
                            ]
                        )

                except Exception as e:
                    logger.error(
                        "[{}] SilenceChecker: failed to queue check-in: {}",
                        self._stream_id,
                        e,
                    )

    async def process_frame(
        self,
        frame: Frame,
        direction: FrameDirection,
    ) -> None:
        await super().process_frame(frame, direction)

        # FIX: was gated on `direction == FrameDirection.UPSTREAM`. This
        # processor sits near the END of the pipeline (after
        # _TerminationProcessor, before the output transport) -- from that
        # position, UserStartedSpeakingFrame and TranscriptionFrame arrive
        # travelling DOWNSTREAM (they originate at the STT service near the
        # START of the pipeline). Gating on UPSTREAM meant on_user_speech()
        # was almost certainly never called, so the silence timer never
        # reset on real user speech. on_user_speech() is idempotent, so
        # checking on either direction is safe.
        if isinstance(frame, UserStartedSpeakingFrame):
            self.on_user_speech()

        elif (
            isinstance(frame, TranscriptionFrame)
            and frame.text.strip()
        ):
            self.on_user_speech()

        # NEW: pause the silence clock for the duration of the bot's own
        # speech, and restart it from zero the moment the bot finishes.
        # Previously nothing told this processor the bot was talking, so
        # an 8-11s bot turn could exceed silence_threshold_secs on its own
        # and fire "Are you still there?" over the bot's own audio.
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_is_speaking = True

        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_is_speaking = False
            self.on_user_speech()

        await self.push_frame(frame, direction)


class _TerminationProcessor(FrameProcessor):
    """
    Detect farewell text without buffering or rewriting normal LLM output.

    This processor handles:
    - Explicit termination leaks from the LLM.
    - Farewell text generated by the LLM.
    - Cancellation when the caller interrupts a pending goodbye.
    """

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
        # NEW: safety_seconds was accepted and stored above but never used
        # anywhere else in this class -- a safety net that was declared
        # but never wired up. If BotStoppedSpeakingFrame never arrived
        # (e.g. a frame-direction mismatch in this pipecat version), a
        # detected farewell would set _waiting_for_bot_stop=True and then
        # wait forever, which is exactly "the call wouldn't cut
        # automatically, had to be done manually." This task is the
        # actual backstop; see _run_safety_timeout below.
        self._safety_task: asyncio.Task | None = None

        self._shutdown_state: dict[str, bool] = {}
        self._end_call_pending: dict[str, bool] = {}

        self.silent_termination_patterns = [
            (
                "end_call_leak",
                re.compile(
                    r"\bend[_\s-]?call\b\.?",
                    re.IGNORECASE,
                ),
            ),
            (
                "tool_narration",
                re.compile(
                    r"\b(call|dial|invok|trigger|activat)\w*\s+"
                    r"(the\s+)?(tool|function|api)\b",
                    re.IGNORECASE,
                ),
            ),
            (
                "explicit_command",
                re.compile(
                    r"\b(end|clos|terminat|finish|hang)\w*\s+"
                    r"(the\s+)?(call|conversation|session|up|tool)\b",
                    re.IGNORECASE,
                ),
            ),
            (
                "disposition_leak",
                re.compile(
                    r"\b(site\s+visit|inventory|lead)\s+"
                    r"(is\s+)?(booked|sent|partial)\b",
                    re.IGNORECASE,
                ),
            ),
            (
                "explicit_tag",
                re.compile(
                    r"\[hangup\]|\[end\]",
                    re.IGNORECASE,
                ),
            ),
        ]

        self.spoken_termination_patterns = [
            re.compile(
                r"\b(goodbye|bye|good day|take care)\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\bhave a (nice|great|good) "
                r"(day|evening|night)\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\btalk to you (later|soon)\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\bthank you for your time\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\bthanks for your time\b",
                re.IGNORECASE,
            ),
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
                await self._force_hangup_fn(
                    "farewell_complete"
                )
                self._provider_hangup_sent = True

            await self.push_frame(
                EndTaskFrame(),
                FrameDirection.UPSTREAM,
            )

        except asyncio.CancelledError:
            return

    async def _schedule_hangup(
        self,
        delay: float,
    ) -> None:
        try:
            await asyncio.sleep(delay)

            if not self._provider_hangup_sent:
                await self._force_hangup_fn(
                    "termination_immediate"
                )
                self._provider_hangup_sent = True

            await self.push_frame(
                EndTaskFrame(),
                FrameDirection.UPSTREAM,
            )

        except asyncio.CancelledError:
            return

    async def _run_safety_timeout(self) -> None:
        """
        NEW: backstop for when the normal BotStoppedSpeakingFrame-triggered
        hangup never fires. self._safety_seconds existed as a constructor
        param before this fix but was never read anywhere -- this is what
        actually wires it up. Without this, a missed or mis-ordered
        BotStoppedSpeakingFrame left the call connected indefinitely after
        a farewell, requiring a manual hangup.
        """
        try:
            await asyncio.sleep(self._safety_seconds)
        except asyncio.CancelledError:
            return

        if self._provider_hangup_sent:
            return

        logger.warning(
            "[{}] Farewell safety timeout ({}s) -- BotStoppedSpeakingFrame "
            "never arrived; forcing hangup anyway",
            self._stream_id,
            self._safety_seconds,
        )

        await self._force_hangup_fn("termination_safety_timeout")
        self._provider_hangup_sent = True

        await self.push_frame(
            EndTaskFrame(),
            FrameDirection.UPSTREAM,
        )

    async def process_frame(
        self,
        frame: Frame,
        direction: FrameDirection,
    ) -> None:

        # A caller interruption cancels a pending farewell.
        if (
            direction == FrameDirection.DOWNSTREAM
            and isinstance(frame, InterruptionFrame)
        ):
            self._waiting_for_bot_stop = False

            if (
                self._hangup_task
                and not self._hangup_task.done()
            ):
                self._hangup_task.cancel()

            self._hangup_task = None

            if (
                self._safety_task
                and not self._safety_task.done()
            ):
                self._safety_task.cancel()

            self._safety_task = None
            self._termination_requested = False
            self._provider_hangup_sent = False

            self._shutdown_state["active"] = False
            self._end_call_pending["active"] = False

            await super().process_frame(
                frame,
                direction,
            )

            await self.push_frame(
                frame,
                direction,
            )

            return

        # Inspect bot speech for explicit termination.
        if (
            direction == FrameDirection.DOWNSTREAM
            and isinstance(
                frame,
                (TextFrame, TTSSpeakFrame),
            )
        ):
            text = getattr(frame, "text", "") or ""

            silent_match = any(
                pattern.search(text)
                for _, pattern in self.silent_termination_patterns
            )

            spoken_match = any(
                pattern.search(text)
                for pattern in self.spoken_termination_patterns
            )

            if silent_match:
                self._termination_requested = True
                self._shutdown_state["active"] = True

                if (
                    self._hangup_task
                    and not self._hangup_task.done()
                ):
                    self._hangup_task.cancel()

                self._hangup_task = asyncio.create_task(
                    self._schedule_hangup(0.5)
                )

                logger.info(
                    "[{}] Silent termination detected; forcing hangup",
                    self._stream_id,
                )

            elif spoken_match:
                self._termination_requested = True
                self._waiting_for_bot_stop = True
                self._shutdown_state["active"] = True

                # NEW: arm the safety-timeout backstop the moment we start
                # waiting for BotStoppedSpeakingFrame. See _run_safety_
                # timeout for why this exists.
                if (
                    self._safety_task
                    and not self._safety_task.done()
                ):
                    self._safety_task.cancel()

                self._safety_task = asyncio.create_task(
                    self._run_safety_timeout()
                )

                logger.info(
                    "[{}] Farewell detected; will hang up after bot stops speaking",
                    self._stream_id,
                )

            await super().process_frame(
                frame,
                direction,
            )

            await self.push_frame(
                frame,
                direction,
            )

            return

        # Once the farewell has actually finished playing, hang up.
        #
        # FIXED: was gated on `direction == FrameDirection.UPSTREAM`. In
        # this same file, _SilenceChecker deliberately does NOT direction-
        # filter Bot(Started|Stopped)SpeakingFrame (see its own comment on
        # why direction filtering broke UserStartedSpeakingFrame handling
        # nearby) -- two processors in the same pipeline making opposite
        # assumptions about which direction this exact frame type travels
        # is the actual root cause of the flakiness reported after the
        # ChatGPT patch: removing the direction check "fixed" the hangup
        # in one run and skipped it in another, because it was never
        # verified which way this frame actually flows in this pipecat
        # version. Matching on frame type alone, regardless of direction,
        # removes that guesswork; the _safety_task above is the backstop
        # if this still somehow doesn't fire.
        if isinstance(frame, BotStoppedSpeakingFrame):
            if (
                self._waiting_for_bot_stop
                and not self._hangup_task
            ):
                logger.info(
                    "[{}] Farewell finished; hanging up in {}s",
                    self._stream_id,
                    self._grace_seconds,
                )

                if (
                    self._safety_task
                    and not self._safety_task.done()
                ):
                    self._safety_task.cancel()

                self._safety_task = None

                self._hangup_task = asyncio.create_task(
                    self._hangup_after_bot_stop()
                )

                self._waiting_for_bot_stop = False

            await super().process_frame(
                frame,
                direction,
            )

            await self.push_frame(
                frame,
                direction,
            )

            return

        await super().process_frame(
            frame,
            direction,
        )

        await self.push_frame(
            frame,
            direction,
        )


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

        self._safety_task = asyncio.create_task(
            self._run_safety_timeout()
        )

        logger.info(
            "[{}] Call ending requested; waiting for closing line",
            self._stream_id,
        )

    def cancel_ending(self, reason: str) -> None:
        if not self._requested:
            return

        logger.info(
            "[{}] Call ending cancelled ({})",
            self._stream_id,
            reason,
        )

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
        await super().process_frame(
            frame,
            direction,
        )

        if direction == FrameDirection.DOWNSTREAM:

            if (
                isinstance(frame, InterruptionFrame)
                and (
                    self._awaiting_closing
                    or self._closing_in_progress
                    or self._in_grace
                )
            ):
                self.cancel_ending(
                    "caller spoke during closing"
                )

            elif (
                isinstance(frame, TTSStartedFrame)
                and self._awaiting_closing
            ):
                self._awaiting_closing = False
                self._closing_in_progress = True

                logger.debug(
                    "[{}] Closing statement started speaking",
                    self._stream_id,
                )

            elif (
                isinstance(frame, TTSStoppedFrame)
                and self._closing_in_progress
            ):
                self._closing_in_progress = False
                self._in_grace = True

                logger.debug(
                    "[{}] Closing statement finished; {}s grace window open",
                    self._stream_id,
                    self._grace_seconds,
                )

                self._cancel_task("_grace_task")

                self._grace_task = asyncio.create_task(
                    self._run_grace_timeout()
                )

        await self.push_frame(
            frame,
            direction,
        )


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
        await super().process_frame(
            frame,
            direction,
        )

        if direction == FrameDirection.DOWNSTREAM:
            before = len(self._context.messages)

            _prune_history(
                self._context.messages,
                self._max_messages,
            )

            after = len(self._context.messages)

            if after < before:
                logger.debug(
                    "[{}] Pruned conversation history {} -> {} messages",
                    self._stream_id,
                    before,
                    after,
                )

        await self.push_frame(
            frame,
            direction,
        )


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
        campaign_data.get("provider_call_id")
        if campaign_data
        else None
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

    # Pre-warm providers and connection pools in parallel with handshake.
    warmup_task = asyncio.create_task(
        warmup_providers(
            config,
            aiohttp_session,
        )
    )

    active = config["active_providers"]

    sample_rate = (
        16000
        if call_type == "web"
        else config["audio"]["sample_rate"]
    )

    provider_construction_task = asyncio.create_task(
        _create_provider_services_in_parallel(
            config,
            active,
            sample_rate=sample_rate,
            aiohttp_session=aiohttp_session,
        )
    )

    # Vobiz handshake.
    if not stream_id:
        try:
            for _ in range(5):
                message = await websocket.receive_text()
                data = json.loads(message)

                candidate_stream_id = (
                    data.get("streamId")
                    or data.get("start", {}).get("streamId")
                )

                candidate_call_id = (
                    data.get("callId")
                    or data.get("start", {}).get("callId")
                )

                if candidate_call_id:
                    provider_call_id = candidate_call_id

                if candidate_stream_id:
                    stream_id = candidate_stream_id
                    break

            if not stream_id:
                stream_id = "unknown"

        except Exception:
            logger.exception(
                "Failed to parse initial Vobiz messages"
            )

            stream_id = "unknown_error"

    call_label = call_id or stream_id or "no-call-id"

    if stream_id and provider_call_id:
        STREAM_PROVIDER_CALL_IDS[stream_id] = provider_call_id

    elif stream_id:
        provider_call_id = STREAM_PROVIDER_CALL_IDS.get(
            stream_id
        )

    conversation_call_type = (
        "outbound"
        if call_type == "web"
        else call_type
    )

    logger.info(
        "[{}] Starting transport={} conversation_type={} call_id={} providers={}",
        stream_id,
        call_type,
        conversation_call_type,
        call_label,
        config.get("active_providers", {}),
    )

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

        auth_id = os.getenv("VOBIZ_AUTH_ID")
        auth_token = os.getenv("VOBIZ_AUTH_TOKEN")

        if not auth_id or not auth_token:
            logger.warning(
                "[{}] Missing Vobiz auth credentials ({})",
                stream_id,
                trigger,
            )
            return

        url = (
            f"https://api.vobiz.ai/api/v1/Account/"
            f"{auth_id}/Call/{provider_call_id}/"
        )

        max_retries = 3
        base_delay = 2.0

        for attempt in range(max_retries):
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
                        "[{}] Vobiz hangup succeeded on attempt {} ({})",
                        stream_id,
                        attempt + 1,
                        trigger,
                    )
                    return

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

                logger.info(
                    "[{}] Retrying Vobiz hangup in {}s...",
                    stream_id,
                    delay,
                )

                await asyncio.sleep(delay)

        logger.error(
            "[{}] Vobiz hangup FAILED after {} attempts ({}). "
            "PSTN call may still be connected and billing!",
            stream_id,
            max_retries,
            trigger,
        )

    # NEW: hoisted above the try block (was previously defined deep inside
    # it) so it's guaranteed to exist for the finally-block safety net
    # below, even if setup fails before reaching that point.
    hangup_state = {
        "done": False,
    }

    async def _force_hangup_and_mark_done(trigger: str) -> None:
        """
        force_provider_hangup wrapper that also marks hangup_state["done"].

        _TerminationProcessor and hard_timeout() both call the hangup
        function directly rather than through _perform_end_of_call_hangup,
        so hangup_state never got set on those paths -- meaning a
        perfectly normal, successful call end still tripped the
        finally-block safety net's warning and fired a redundant second
        hangup attempt. This wrapper closes that gap without touching
        either class's working cancellation flow (EndTaskFrame push /
        task.cancel() timing stays exactly as it was).
        """
        hangup_state["done"] = True
        await force_provider_hangup(trigger)

    log_handler = logger.add(
        f"logs/call_{call_label}_{stream_id}.log",
        level="DEBUG",
    )

    call_metrics = None

    try:
        serializer = (
            WebPCMFrameSerializer(
                stream_id=stream_id,
            )
            if call_type == "web"
            else VobizFrameSerializer(
                stream_id=stream_id,
                sample_rate=8000,
            )
        )

        transport = FastAPIWebsocketTransport(
            websocket=websocket,
            params=FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                add_wav_header=False,
                vad_enabled=False,
                vad_audio_passthrough=(
                    active.get("stt") == "deepgram"
                ),
                serializer=serializer,
            ),
        )

        # Wait for provider construction and the generic warmup pass.
        stt, llm, tts = await provider_construction_task
        await warmup_task

        # Build the system prompt now (cheap, local, no I/O) so we can fire
        # a REAL LLM cache-warm immediately -- through the actual client
        # object GroqLLMService will use for the real completion, with the
        # exact prompt that will be sent. This replaces the old
        # aiohttp-session-based warmup, which fired over a connection pool
        # the SDK's own client never touched and empirically never moved
        # turn-1 latency across multiple test calls (still ~1.7-2.0s TTFB
        # every time, warmup or not). This one hits the same client, same
        # prompt, same provider-side prefix cache the real turn will need.
        system_prompt, customer_context = build_system_prompt(
            conversation_call_type,
            config,
            campaign_data,
        )

        # NEW: strip the per-call "Current time: ..." suffix before this
        # prompt is used anywhere. Groq's prompt cache requires an exact,
        # character-for-character prefix match across requests -- a
        # wall-clock timestamp baked into the tail of the system prompt
        # guarantees every request has a distinct "stable" prefix, which
        # defeats caching regardless of whether it's enabled for this
        # model. See _strip_dynamic_time_suffix() docstring above.
        system_prompt = _strip_dynamic_time_suffix(system_prompt)

        # NEW: moved up from further below (previously computed only after
        # call_metrics/language_state/end_call/etc. were set up). This is
        # now needed *before* the cache pre-warm fires, because the
        # pre-warm request must include the same tool schema the real
        # turns send -- see the comment on _warm_llm_context_cache below
        # for why an incomplete prefix silently poisons a cache key that
        # never gets reused.
        llm_tools_raw = (
            config.get("providers", {})
            .get("llm", {})
            .get(active["llm"], {})
            .get("params", {})
            .get("tools", [])
        )

        llm_tools_raw = [
            raw
            for raw in llm_tools_raw
            if not (
                isinstance(raw, dict)
                and raw.get(
                    "function",
                    {},
                ).get(
                    "name"
                )
                in ("set_conversation_language", "end_call")
            )
        ]

        async def _warm_llm_context_cache() -> None:
            """
            Fire-and-forget pre-warm of the LLM provider's request cache.

            CHANGED: now includes `tools=llm_tools_raw` in the warm-up
            call. Per Groq's docs, the entire request prefix is cacheable
            -- messages AND tool definitions together -- and a cache hit
            requires an exact match of that whole prefix. The previous
            version warmed `messages=[system]` with no tools param, while
            every real turn sends `messages=[system, ...] + tools=[...]`.
            Those are two different prefixes, so the previous pre-warm was
            priming a cache entry no real request could ever hit -- which
            is a much more likely explanation for the "pre-warm failed"
            log line on every single call than prompt length or Groq-side
            unavailability.

            CHANGED: now logs the actual response usage, including
            cached_tokens if present, instead of silently discarding it.
            This is the only way to know for certain whether prompt
            caching is even live for this model (qwen/qwen3.8-27b) --
            Groq's caching rollout is per-model and, as of the last public
            community request thread, was not yet confirmed for the
            Qwen3 line. Don't assume the win; check this log line.

            CHANGED: exception logging now includes the traceback
            (`logger.opt(exception=True)`) instead of a bare debug message
            with no diagnostic content. Every call in the provided logs
            hit this except-block and there was no way to tell why -- and
            the traceback that surfaced (`AttributeError: 'GroqLLMService'
            object has no attribute 'client'`) is exactly why: this
            function was reaching into a private attribute on the pipecat
            service object that doesn't exist under that name. It crashed
            on that line on every single call, before ever reaching Groq,
            which means the cache-verification logging below has never
            once actually run.

            CHANGED: no longer touches `llm`'s internals at all. Rather
            than guess at the correct private attribute a second time,
            this builds its own short-lived AsyncOpenAI client pointed at
            the same base_url pipecat's own GroqLLMService logs using
            ("Creating Groq client with api https://api.groq.com/openai/v1")
            and the same GROQ_API_KEY. Server-side prompt caching keys off
            the request content, not which client object sent it, so this
            is functionally equivalent to reusing the internal client and
            can't break again the same way if pipecat's internals change.

            FIXED: warm_kwargs previously sent messages=[system] with no
            user-role message. Groq's chat template requires at least one
            user message to render at all -- see the openai.BadRequestError
            in the Sept 1 log ("failed to render text output: ... raise_
            exception: No user query found in messages"). That 400 fired
            on every single call, meaning this pre-warm has never once
            succeeded. Adding a short, inert placeholder user message
            fixes the request shape; it costs nothing extra since
            max_tokens=1 discards the completion anyway.
            """
            try:
                model = (
                    config.get("providers", {})
                    .get("llm", {})
                    .get(active["llm"], {})
                    .get("params", {})
                    .get("model")
                )

                warm_kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": "Hello"},
                    ],
                    "max_tokens": 1,
                }

                if llm_tools_raw:
                    warm_kwargs["tools"] = llm_tools_raw

                groq_api_key = os.getenv("GROQ_API_KEY")

                if not groq_api_key:
                    raise RuntimeError(
                        "GROQ_API_KEY not set; cannot pre-warm"
                    )

                async with AsyncOpenAI(
                    api_key=groq_api_key,
                    base_url="https://api.groq.com/openai/v1",
                ) as warm_client:
                    response = await warm_client.chat.completions.create(
                        **warm_kwargs
                    )

                usage = getattr(response, "usage", None)
                cached_tokens = 0

                if usage is not None:
                    details = getattr(usage, "prompt_tokens_details", None)
                    cached_tokens = getattr(details, "cached_tokens", 0) or 0

                if cached_tokens > 0:
                    logger.info(
                        "[{}] LLM context cache pre-warm HIT: "
                        "{} tokens reused",
                        stream_id,
                        cached_tokens,
                    )
                else:
                    logger.info(
                        "[{}] LLM context cache pre-warm sent "
                        "(cached_tokens=0 -- cache miss, or caching not "
                        "yet enabled for this model)",
                        stream_id,
                    )

            except Exception:
                logger.opt(exception=True).debug(
                    "[{}] LLM context cache pre-warm failed",
                    stream_id,
                )

        # Fire and forget -- runs concurrently with the rest of pipeline
        # setup, the greeting playback, and the caller's think-time before
        # replying, so it has several seconds' head start on the real
        # first completion.
        asyncio.create_task(_warm_llm_context_cache())

        call_metrics = CallMetricsCollector(
            call_id=stream_id,
            stt_provider=active["stt"],
            llm_provider=active["llm"],
            tts_provider=active["tts"],
        )

        language_state = LanguageState(
            sustained_switch_turns=config.get(
                "language",
                {},
            ).get(
                "sustained_switch_turns",
                2,
            ),
        )

        shutdown_state = {
            "active": False,
        }

        end_call_pending = {
            "active": False,
        }

        # FIX:
        # shutdown_state["active"] means "we are in the closing phase".
        # It must NOT mean "the hangup has already been performed".
        #
        # Those are separate states.
        async def _perform_end_of_call_hangup() -> None:
            if hangup_state["done"]:
                return

            hangup_state["done"] = True
            shutdown_state["active"] = True

            logger.info(
                "[{}] Ending call now (closing statement completed)",
                stream_id,
            )

            await force_provider_hangup(
                "call_end_coordinator"
            )

            await task.cancel()

        termination_processor = _TerminationProcessor(
            stream_id=stream_id,
            on_hangup=_perform_end_of_call_hangup,
            # FIX: was force_hangup_fn=force_provider_hangup directly. This
            # class's own hangup paths call force_hangup_fn without ever
            # touching hangup_state, so a perfectly normal, successful call
            # end still tripped the finally-block safety net's "hangup was
            # never confirmed done" warning and fired a redundant second
            # hangup attempt. The wrapper below marks hangup_state["done"]
            # first, same as _perform_end_of_call_hangup does, without
            # touching this class's working EndTaskFrame/cancellation flow.
            force_hangup_fn=_force_hangup_and_mark_done,
            grace_seconds=1.5,
            safety_seconds=15.0,
        )

        termination_processor.bind_state(
            shutdown_state=shutdown_state,
            end_call_pending=end_call_pending,
        )

        call_end_coordinator = _CallEndCoordinator(
            stream_id=stream_id,
            on_hangup=_perform_end_of_call_hangup,
            grace_seconds=1.5,
            safety_seconds=15.0,
        )

        interruption_audio_gate = _InterruptionAudioGate(
            stream_id=stream_id,
            on_interruption=(
                serializer.on_interruption
                if hasattr(
                    serializer,
                    "on_interruption",
                )
                else None
            ),
        )

        def _latest_user_utterance() -> str:
            for message in reversed(messages):
                if message.get("role") == "user":
                    content = message.get(
                        "content",
                        "",
                    )

                    return (
                        content.strip()
                        if isinstance(content, str)
                        else ""
                    )

            return ""

        end_call_blocked_this_turn = {
            "active": False,
        }

        async def end_call(
            params: FunctionCallParams,
        ):
            tool_call_started_at = time.monotonic()
            latest_user_text = _latest_user_utterance()

            if not latest_user_text:
                logger.warning(
                    "[{}] Rejected end_call before first caller utterance",
                    stream_id,
                )

                await params.result_callback(
                    {
                        "status": "rejected",
                        "reason": "no caller utterance yet",
                    },
                    properties=FunctionCallResultProperties(
                        run_llm=True,
                    ),
                )

                return

            # HARD GATE:
            # The LLM may request end_call, but it cannot authorize termination.
            # The current bot.py has no application-level "objective fulfilled"
            # state, so objective completion is intentionally not inferred
            # from arbitrary caller utterances. Only an explicit caller
            # termination/decline can authorize this terminal action.
            if not _caller_explicitly_ended(latest_user_text):
                logger.warning(
                    "[{}] Blocked end_call: caller has not explicitly "
                    "ended or declined. latest_user_utterance={!r}",
                    stream_id,
                    latest_user_text,
                )

                already_blocked = end_call_blocked_this_turn["active"]
                end_call_blocked_this_turn["active"] = True

                await params.result_callback(
                    {
                        "status": "rejected",
                        "reason": (
                            "terminal condition not fulfilled; "
                            "continue the conversation"
                        ),
                    },
                    properties=FunctionCallResultProperties(
                        run_llm=not already_blocked,
                    ),
                )

                call_metrics.record_tool_call(
                    "end_call",
                    success=False,
                    latency_ms=(
                        time.monotonic()
                        - tool_call_started_at
                    ) * 1000,
                )

                return

            if call_end_coordinator.is_ending:
                logger.info(
                    "[{}] Ignored duplicate end_call while already ending",
                    stream_id,
                )

                await params.result_callback(
                    {
                        "status": "already_ending",
                    },
                    properties=FunctionCallResultProperties(
                        run_llm=False,
                    ),
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
                # FIX: was run_llm=False. That silently discarded the "say"
                # instruction below -- nothing ever triggered a follow-up
                # completion to actually speak it. It only appeared to work
                # when the model happened to also emit spoken text in the
                # SAME turn as the tool call (pure luck, not guaranteed).
                # When it didn't (confirmed in logs: a turn with only 14
                # completion tokens -- just the bare tool call, no text),
                # the result was 10+ seconds of dead air, the caller
                # re-prompting, and the ending sequence getting cancelled
                # as "caller spoke during closing". run_llm=True guarantees
                # a follow-up completion always runs, so the model always
                # reads its own "say" instruction and always speaks the
                # closing line. Safe against a duplicate end_call in that
                # follow-up turn: the "already_ending" branch above catches
                # it and returns run_llm=False, so there's no loop risk.
                await params.result_callback(
                    {
                        "status": "ending",
                        "say": (
                            "Speak one brief, warm closing line now "
                            "that fits what was actually discussed "
                            "on this call, thank the caller, and do not "
                            "call any tool this turn."
                        ),
                    },
                    properties=FunctionCallResultProperties(
                        run_llm=True,
                    ),
                )

            except Exception:
                call_end_coordinator.cancel_ending(
                    "end_call setup raised an exception"
                )

                call_metrics.record_tool_call(
                    "end_call",
                    success=False,
                    latency_ms=(
                        time.monotonic()
                        - tool_call_started_at
                    ) * 1000,
                )

                raise

            else:
                call_metrics.record_tool_call(
                    "end_call",
                    success=True,
                    latency_ms=(
                        time.monotonic()
                        - tool_call_started_at
                    ) * 1000,
                )

        llm.register_function(
            "end_call",
            end_call,
            cancel_on_interruption=True,
        )

        # system_prompt / customer_context were already built above (and
        # the dynamic time suffix stripped), so the cache-warm task could
        # fire as early as possible.
        #
        # BYPASS-LLM-FOR-GREETING NOTE: `messages` below is the same list
        # object LLMContext will wrap a few lines down. The greeting text
        # is appended to it directly inside on_client_connected() further
        # below via `messages.append({"role": "assistant", ...})`, and is
        # spoken via TTSSpeakFrame(append_to_context=False) rather than by
        # queuing an LLMContextFrame -- so the greeting never triggers an
        # LLM completion. The context object still knows the greeting was
        # said (via the manual append), so when the caller replies, the
        # model sees the full turn history without having had to generate
        # the greeting itself. This was already the design in this file;
        # nothing changed here, called out explicitly since it's exactly
        # the "bypass LLM for greeting" pattern.
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]

        if customer_context:
            messages.append(
                {
                    "role": "system",
                    "content": customer_context,
                }
            )

        # llm_tools_raw computed earlier (see above, before the cache
        # pre-warm fires) so the pre-warm request's tool schema exactly
        # matches what real turns send.
        llm_tools = [
            FunctionSchema(
                name=raw["function"]["name"],
                description=raw["function"].get(
                    "description",
                    "",
                ),
                properties=(
                    raw["function"]
                    .get("parameters", {})
                    .get("properties", {})
                ),
                required=(
                    raw["function"]
                    .get("parameters", {})
                    .get("required", [])
                ),
            )
            for raw in llm_tools_raw
        ]

        context = LLMContext(
            messages,
            tools=llm_tools,
        )

        turn_config = config.get(
            "turn_management",
            {},
        )

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
                    turn_config.get(
                        "user_turn_stop_timeout",
                        1.0,
                    )
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
            max_messages=config.get(
                "max_conversation_messages",
                20,
            ),
            stream_id=stream_id,
        )

        spoken_text_guard = _SpokenTextGuard()

        silence_checker = _SilenceChecker(
            stream_id=stream_id,
            task=None,
            context_aggregator_user=context_aggregator.user(),
            silence_threshold_secs=5.0,
            check_in_message="Are you still there?",
        )

        # REMOVED: _FirstOutboundResponse.
        #
        # It intercepted the caller's first "yes/hi/hello" and pushed a
        # canned TTSSpeakFrame directly, skipping the LLM -- but it never
        # forwarded the underlying TranscriptionFrame downstream. The
        # STT-driven turn-stop signal fired anyway (independent of what
        # this processor swallowed) partway through that canned reply's
        # playback, which bumped the audio generation id mid-utterance.
        # _InterruptionAudioGate then treated the reply's own still-
        # streaming audio as stale and dropped it -- producing exactly
        # the "starts talking, goes silent, caller has to prompt it to
        # continue" bug. It also cost more than it saved: it only
        # deferred the cold LLM call to the next turn rather than
        # eliminating it, while silently dropping the caller's utterance
        # from conversation history. Removed entirely; all replies now
        # go through the normal context_aggregator -> llm -> tts path.

        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                language_observer,
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
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
        )

        silence_checker.set_task(task)

        # Deterministic outbound opening.
        #
        # There is intentionally NO first-turn word classifier here.
        # The application speaks the opening immediately, and the caller's
        # response is handled normally by STT -> turn manager -> LLM.
        _greeting_sent = {
            "done": False,
        }

        @transport.event_handler(
            "on_client_connected"
        )
        async def on_client_connected(
            transport,
            client,
        ):
            if _greeting_sent["done"]:
                logger.debug(
                    "[{}] Greeting already sent, skipping",
                    stream_id,
                )
                return

            logger.info(
                "[{}] Client connected, sending outbound greeting",
                stream_id,
            )

            customer_name = None

            if campaign_data:
                customer_name = campaign_data.get(
                    "customer_name"
                )

            if (
                not customer_name
                and call_type == "web"
            ):
                customer_name = config.get(
                    "test_customer_name"
                )

            if conversation_call_type == "outbound":
                if customer_name:
                    greeting_template = config.get(
                        "greeting_outbound",
                        "Hi, am I speaking with {customer_name}?",
                    )

                    greeting = str(
                        greeting_template
                    ).replace(
                        "{customer_name}",
                        customer_name,
                    )

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
                        config.get(
                            "greeting_inbound",
                            "Hi, how can I help you?",
                        ),
                    )
                )

            # Deterministic outbound opening.
            if (
                conversation_call_type == "outbound"
                and customer_name
            ):
                greeting = (
                    f"Hi, am I speaking with "
                    f"{customer_name}?"
                )

            elif conversation_call_type == "outbound":
                greeting = str(
                    config.get(
                        "greeting_outbound_no_name",
                        "Hi, is this a good time to talk?",
                    )
                )

            # BYPASS LLM FOR GREETING: the greeting text is appended
            # directly to the shared `messages` list (which backs
            # `context`) as an assistant turn, and spoken via
            # TTSSpeakFrame below with append_to_context=False so it is
            # not spoken twice. No LLMContextFrame is queued and no LLM
            # completion is triggered to produce this text -- it goes
            # straight to TTS. The context stays accurate for the
            # caller's first reply because the message list already has
            # this turn in it by the time that reply is processed.
            messages.append(
                {
                    "role": "assistant",
                    "content": greeting,
                }
            )

            interruption_audio_gate.next_generation()

            if hasattr(
                serializer,
                "next_generation",
            ):
                serializer.next_generation()

            _greeting_sent["done"] = True

            await task.queue_frames(
                [
                    TTSSpeakFrame(
                        text=greeting,
                        append_to_context=False,
                    ),
                ]
            )

            async def hard_timeout():
                await asyncio.sleep(
                    max_duration
                )

                # FIX: was checking/setting shutdown_state["active"] and
                # calling force_provider_hangup directly. hangup_state is
                # the authoritative "has hangup actually happened" flag;
                # this now matches the same guard-then-hangup pattern used
                # everywhere else, so a hard timeout that races with a
                # normal ending doesn't fire a redundant second hangup.
                if hangup_state["done"]:
                    return

                logger.warning(
                    "[{}] Hard call timeout",
                    stream_id,
                )

                await _force_hangup_and_mark_done(
                    "hard_timeout"
                )

                await task.cancel()

            asyncio.create_task(
                hard_timeout()
            )

        @context_aggregator.user().event_handler(
            "on_user_turn_stopped"
        )
        async def on_user_turn_stopped(
            processor,
            strategy,
            *args,
            **kwargs,
        ):
            end_call_blocked_this_turn["active"] = False

            gen_id = (
                interruption_audio_gate.next_generation()
            )

            if hasattr(
                serializer,
                "next_generation",
            ):
                serializer.next_generation()

            logger.debug(
                "[{}] Advanced audio generation context to id={}",
                stream_id,
                gen_id,
            )

        @transport.event_handler(
            "on_client_disconnected"
        )
        async def on_client_disconnected(
            transport,
            client,
        ):
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
        STREAM_PROVIDER_CALL_IDS.pop(
            stream_id,
            None,
        )

        if call_metrics is not None:
            call_metrics.finalize()

        # NEW: guaranteed last-resort hangup. Every normal path
        # (call_end_coordinator finishing a closing line, the hard
        # timeout, a client disconnect) already sets hangup_state["done"]
        # via _perform_end_of_call_hangup(). If we reach here and that
        # never happened -- a crash mid-pipeline, an exception before
        # the coordinator was even wired up, anything unforeseen -- this
        # is the final backstop that ensures a PSTN call is never left
        # connected and billing. Safe to call even with no provider_call_id
        # (force_provider_hangup no-ops and logs a warning in that case).
        if not hangup_state["done"]:
            logger.warning(
                "[{}] Finally-block safety net: hangup was never "
                "confirmed done through the normal paths; forcing one "
                "last attempt now.",
                stream_id,
            )

            try:
                await force_provider_hangup("finally_safety_net")

            except Exception:
                logger.exception(
                    "[{}] Finally-block safety-net hangup attempt failed",
                    stream_id,
                )

            hangup_state["done"] = True

        if not aiohttp_session.closed:
            await aiohttp_session.close()

        logger.remove(
            log_handler
        )