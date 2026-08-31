"""
Production voicebot orchestration.

The bot deliberately does NOT contain a hand-written language classifier.

Language decisions come from:
1. multilingual STT metadata for automatic detection;
2. narrow local detection of explicit caller language requests.

LanguageState is per-call. TTS language changes use Pipecat's
runtime TTSUpdateSettingsFrame so native and custom TTS providers follow the
same runtime-settings path.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
import time
from importlib import import_module
from typing import Any, Optional

import aiohttp
import httpx
from dotenv import load_dotenv
from fastapi import WebSocket
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.frames.frames import (
    FunctionCallResultProperties,
    Frame,
    LLMContextFrame,
    TTSUpdateSettingsFrame,
    TranscriptionFrame,
    TextFrame,
    TTSSpeakFrame,
)
from pipecat.services.llm_service import FunctionCallParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.turns.user_start import (
    VADUserTurnStartStrategy,
    TranscriptionUserTurnStartStrategy,
)
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
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


class LanguageObserver(FrameProcessor):
    """Keep LLM context and TTS synchronized with the caller's language.

    Two paths can change language:
    1. reliable STT language metadata;
    2. an explicit language request in the caller's utterance.

    Explicit language requests are handled locally so language changes do not
    depend on an LLM tool decision.
    """

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
        messages: list[dict[str, Any]],
    ) -> None:
        super().__init__()
        self._stream_id = stream_id
        self._language_state = language_state
        self._tts = tts
        self._tts_provider = tts_provider
        self._messages = messages
        self._base_system_prompt = str(
            messages[0].get("content", "")
        ) if messages else ""

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
            transcript = self._extract_transcript(frame)
            explicit_language = self._detect_explicit_language_request(
                transcript
            )

            language, probability = self._extract_language(frame)

            # An explicit caller request is authoritative. This is intentionally
            # resolved before STT metadata because the STT may label an English
            # sentence such as "please speak in Telugu" as English.
            requested_language = explicit_language or language

            if requested_language:
                old_language = self._language_state.current_language

                if explicit_language:
                    new_language, switched = (
                        self._language_state.set_explicit(
                            explicit_language,
                            "explicit_request",
                        )
                    )
                    reason = "explicit_request"
                else:
                    new_language, switched = (
                        self._language_state.observe_stt(
                            language,
                            probability,
                        )
                    )
                    reason = self._language_state.switch_reason

                logger.info(
                    "[{}] language observation old={} detected={} "
                    "confidence={} current={} switched={} reason={} "
                    "transcript={!r}",
                    self._stream_id,
                    old_language,
                    requested_language,
                    probability,
                    new_language,
                    switched,
                    reason,
                    transcript,
                )

                if switched:
                    await self._apply_language_change(
                        new_language,
                        reason,
                    )

            elif language:
                logger.debug(
                    "[{}] Unsupported/uncertain language metadata "
                    "language={!r} transcript={!r}",
                    self._stream_id,
                    language,
                    transcript,
                )

        await self.push_frame(
            frame,
            direction,
        )

    @staticmethod
    def _extract_transcript(frame: TranscriptionFrame) -> str:
        for attr in ("text", "transcript"):
            value = getattr(frame, attr, None)
            if isinstance(value, str) and value.strip():
                return value.strip()

        result = getattr(frame, "result", None)
        if isinstance(result, dict):
            for container in (
                result,
                result.get("data") if isinstance(result.get("data"), dict) else {},
            ):
                value = (
                    container.get("transcript")
                    or container.get("text")
                )
                if isinstance(value, str) and value.strip():
                    return value.strip()

        return ""

    @staticmethod
    def _extract_language(
        frame: TranscriptionFrame,
    ) -> tuple[Optional[str], Optional[float]]:
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
            data = result.get("data")
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
                    if container.get("language_probability") is not None
                    else container.get("language_confidence")
                )
                if raw_probability is None:
                    raw_probability = container.get("confidence")

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

    @classmethod
    def _detect_explicit_language_request(
        cls,
        transcript: str,
    ) -> Optional[LanguageCode]:
        """Detect an explicit request to speak/respond in a language.

        This is deliberately narrow. It catches direct language-control
        utterances without trying to classify ordinary conversation text.
        """
        text = (transcript or "").strip()
        if not text:
            return None

        lowered = text.lower()

        language_terms = {
            "en": (
                "english",
                "इंग्लिश",
                "अंग्रेज़ी",
                "अंग्रेजी",
            ),
            "hi": (
                "hindi",
                "हिंदी",
                "हिन्दी",
            ),
            "te": (
                "telugu",
                "తెలుగు",
                "తెలుగులో",
            ),
            "ta": (
                "tamil",
                "தமிழ்",
                "தமிழில்",
            ),
        }

        control_terms = (
            "speak",
            "talk",
            "reply",
            "respond",
            "answer",
            "continue",
            "switch",
            "language",
            "can you speak",
            "please speak",
            "speak in",
            "speak to me in",
            "talk in",
            "reply in",
            "answer in",
            "बोल",
            "बोलिए",
            "बोलो",
            "हिंदी में",
            "तेलुगु में",
            "தமிழில்",
            "తెలుగులో",
        )

        # Exact language-only replies such as "Telugu" are also treated as an
        # explicit selection.
        for language_code, terms in language_terms.items():
            if lowered in terms or text in terms:
                return language_code  # type: ignore[return-value]

        has_control_term = any(
            term in lowered or term in text
            for term in control_terms
        )
        if not has_control_term:
            return None

        for language_code, terms in language_terms.items():
            if any(
                term in lowered or term in text
                for term in terms
            ):
                return language_code  # type: ignore[return-value]

        return None

    async def _apply_language_change(
        self,
        language: LanguageCode,
        reason: str,
    ) -> None:
        locale = LANGUAGE_LOCALES[language]
        label = self._LANGUAGE_LABELS[language]

        # Make the current output language explicit to the LLM. This is a
        # context-level invariant, not a conversational message, so the model
        # cannot mistake the instruction for something the caller said.
        if self._messages:
            self._messages[0]["content"] = (
                f"{self._base_system_prompt}\n\n"
                f"CURRENT RESPONSE LANGUAGE: {label} ({locale}).\n"
                f"Respond in {label} for this turn and subsequent turns "
                f"until the caller clearly changes language."
            )

        # Use Pipecat's supported runtime-settings mechanism instead of
        # expecting provider-specific set_language() methods. Sarvam's native
        # TTS service exposes language through TTS settings.
        await self.push_frame(
            TTSUpdateSettingsFrame(
                settings={
                    "language": locale,
                },
            ),
            FrameDirection.DOWNSTREAM,
        )

        logger.info(
            "[{}] Language applied language={} locale={} reason={} "
            "tts_provider={}",
            self._stream_id,
            label,
            locale,
            reason,
            self._tts_provider,
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

        # `tools` is consumed separately — run_bot() reads it directly from
        # config to build the OpenAILLMContext (see llm_tools below in this
        # file). No Pipecat service constructor accepts a `tools` kwarg, so
        # it must never be forwarded to the service itself.
        params.pop("tools", None)

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
                # FIXED: this used to do `Settings(**params)` with every
                # config.yaml param dumped in unfiltered. Native Pipecat
                # services split configuration between top-level __init__
                # kwargs (connection-level things: sample_rate, mode,
                # keepalive_timeout, encoding...) and the Settings dataclass
                # (runtime-updatable things: model, language, voice,
                # temperature...). Passing a top-level-only field into
                # Settings(**params) raises TypeError immediately — verified
                # empirically for SarvamSTTService's `mode`,
                # DeepgramSTTService's `sample_rate`/`encoding`/`channels`,
                # and (via the `tools` pop above) every OpenAI-compatible
                # LLM service. This was breaking the CURRENT active default
                # config (sarvam STT, sarvam LLM) before any provider
                # changes made today — every call would have failed at
                # service construction, before processing a single turn.
                #
                # Fix: introspect the real constructor and route each
                # param to wherever it actually belongs.
                init_signature = inspect.signature(
                    service_class.__init__
                )
                top_level_names = {
                    name
                    for name in init_signature.parameters
                    if name
                    not in (
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
                }
                settings_params = {
                    key: value
                    for key, value in params.items()
                    if key not in top_level_names
                }

                kwargs.update(top_level_params)

                if settings_params:
                    # FIXED (critical): this used to call
                    # `settings_cls(**settings_params)` directly. Pipecat's
                    # Settings dataclasses only declare fields for the
                    # options Pipecat itself knows about — anything
                    # provider-specific that isn't a declared field (e.g.
                    # Groq/qwen3's `reasoning_format` / `reasoning_effort`)
                    # raises TypeError immediately on construction, which
                    # means the ENTIRE service fails to build and the call
                    # never starts. `from_mapping()` is Settings' own
                    # documented classmethod for exactly this: known fields
                    # go where they belong, anything unrecognized lands in
                    # the settings object's `.extra` dict, which Pipecat's
                    # OpenAI-compatible base client merges straight into the
                    # outgoing API request body. Verified directly against
                    # the installed pipecat-ai package: constructing
                    # GroqLLMService.Settings(reasoning_format="hidden")
                    # raises TypeError; .from_mapping({"reasoning_format":
                    # "hidden"}) does not, and correctly places it in
                    # .extra.
                    kwargs["settings"] = settings_cls.from_mapping(
                        settings_params
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


async def warmup_providers(config: dict) -> None:
    """Pay the one-time provider cold-start cost at process startup instead
    of on whichever call happens to arrive first.

    LATENCY FIX (part of issue 2, "big gap before the bot speaks"): the call
    logs show ~2.1s just constructing the Groq client
    (`Creating llm provider=groq` -> `Creating Groq client with api ...`)
    plus ~250ms loading the Silero VAD model, all landing on the very first
    call served by a fresh worker process — the class imports underneath
    `ServiceFactory._import_class()` (which pulls in the groq/openai/httpx
    module chain) and the VAD model weights are only genuinely expensive the
    first time, then cached by Python/torch for the rest of the process's
    life. The old code paid that cost lazily, inside `run_bot()`, on the
    first real caller. This pays it eagerly, before anyone can call in.

    Wire this into your ASGI app's startup, e.g. in main.py:

        from bot import warmup_providers

        @app.on_event("startup")
        async def _startup():
            await warmup_providers(config)

    (or the equivalent in a `lifespan` context manager, depending on your
    FastAPI version). This does NOT construct real per-call STT/LLM/TTS
    client instances — Sarvam/Deepgram's websocket-backed services are bound
    to one call's audio stream and are not safe to share across concurrent
    calls, so real construction still has to happen per call via
    `_create_provider_services_in_parallel()` below. This only forces the
    underlying classes (and the VAD model) to be resident in memory first.
    """
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
            await asyncio.to_thread(
                ServiceFactory._import_class,
                class_path,
            )
        except Exception:
            logger.exception(
                "Warmup: failed to import {}",
                class_path,
            )

    try:
        await asyncio.to_thread(SileroVADAnalyzer)
    except Exception:
        logger.exception("Warmup: failed to prime Silero VAD")

    logger.info(
        "Warmup complete: {} provider classes imported",
        len(class_paths),
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


class _SpokenTextGuard(FrameProcessor):
    """Prevent internal orchestration text from reaching TTS.

    IMPORTANT: this runs on raw, per-token streaming deltas from the LLM
    (pipeline order is `llm -> spoken_text_guard -> tts`), not on complete
    sentences. That constrains what this guard is allowed to do:

    - It must never drop whitespace-only or punctuation-only deltas.
      Those are normal, expected pieces of a streamed utterance (a comma
      or a space is frequently its own delta), and TTS's own downstream
      text aggregator depends on receiving *all* of them to reconstruct
      correct spacing and pauses. A previous version of this guard treated
      "no alphanumeric characters in this delta" as reason to drop it,
      which silently deleted every space, comma, period, and question
      mark the model streamed. That is what was producing run-on,
      glued-together text like "have2 and3 BHK" / "late2027" reaching
      TTS, and it was *worse* for Hindi: Devanagari vowel signs and other
      combining marks (matras, nukta, candrabindu, etc.) are Unicode
      combining characters, which Python's `str.isalnum()` reports as
      non-alphanumeric even though they're linguistically part of the
      word — so mid-word vowel sounds were being deleted from Hindi
      output. Do not reintroduce an alnum-based filter here.
    - The phrase/regex leak checks below are still safe to run per-delta:
      a 1-3 character streaming fragment can never *contain* a full
      leaked phrase like "you just called this person", so those checks
      only ever fire once enough of a real leak has streamed through in a
      single delta (which is the case they're meant to catch).
    """

    _INTERNAL_PHRASES = (
        "you just called this person",
        "the caller just connected",
        "start with the opening from the campaign script",
        "greet them briefly and ask how you can help",
    )

    _INTERNAL_REGEXES = (
        r"^\s*(user|assistant|system|tool)\s*:",
        r"\b(call|invoke|trigger|activate)\s+(the\s+)?[\w-]+\s+(tool|function|api)\b",
    )

    @staticmethod
    def _text(frame: Frame) -> str:
        value = getattr(frame, "text", None)
        # Deliberately NOT stripped — see class docstring. Stripping a
        # whitespace-only delta down to "" and then treating "" as
        # droppable was how inter-word spacing was getting eaten.
        return value if isinstance(value, str) else ""

    @classmethod
    def _is_blocked(cls, text: str) -> bool:
        # Only a genuinely zero-length delta is unsafe to forward. A
        # single space, comma, period, or combining mark is legitimate
        # content and must pass through untouched.
        if text == "":
            return True

        lowered = text.lower()

        if any(phrase in lowered for phrase in cls._INTERNAL_PHRASES):
            return True

        if any(
            re.search(pattern, text, re.IGNORECASE)
            for pattern in cls._INTERNAL_REGEXES
        ):
            return True

        return False

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

            if self._is_blocked(text):
                logger.warning(
                    "SpokenTextGuard blocked unsafe/empty TTS text={!r}",
                    text,
                )
                return

        await self.push_frame(frame, direction)


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
    """Construct independent provider objects concurrently.

    Provider constructors are synchronous and mostly perform imports/client setup.
    Creating STT, LLM and TTS sequentially makes their cold-start costs add up.
    They do not share mutable per-call state, so construction can safely happen
    in worker threads before the PipelineTask starts.
    """
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
        campaign_data.get(
            "provider_call_id"
        )
        if campaign_data
        else None
    )

    # --------------------------------------------------------
    # LATENCY FIX (issue 2, "big gap before the bot first speaks"):
    #
    # Neither the aiohttp session nor `_create_provider_services_in_parallel`
    # depends on stream_id/call_id — only on `call_type` and `config`, both
    # already available as function arguments. The previous version created
    # the aiohttp session and awaited provider construction only AFTER the
    # Vobiz handshake loop below finished, which made handshake wait time
    # and provider cold-start time fully additive (serial), regardless of
    # whether the caller said anything. Starting construction here lets it
    # run concurrently with the handshake wait instead, so its ~2s+
    # cold-start cost (see `warmup_providers()` above for the bigger
    # structural fix — pre-warming at process startup) at least overlaps
    # with, rather than stacks on top of, the handshake round-trip. This
    # helps every call, not just the first one after a deploy.
    # --------------------------------------------------------
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

    # The browser/web transport is a test harness for the same outbound
    # conversation. Keep the conversational semantics identical to Vobiz
    # outbound calls. Audio transport remains "web"; only prompt/greeting
    # semantics use the outbound campaign.
    conversation_call_type = (
        "outbound" if call_type == "web" else call_type
    )

    logger.info(
        "[{}] Starting transport={} conversation_type={} call_id={} providers={}",
        stream_id,
        call_type,
        conversation_call_type,
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

    # aiohttp_session was already created above, before the handshake, so
    # its construction overlaps with the handshake wait instead of
    # following it.
    call_metrics = None

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
                        0.6,
                    ),
                    confidence=vad_config.get(
                        "start_threshold",
                        0.6,
                    ),
                )
            )

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

        # Provider construction was kicked off before the handshake above —
        # await it here. In the common case it has already finished by now.
        stt, llm, tts = await provider_construction_task

        call_metrics = CallMetricsCollector(
            call_id=stream_id,
            stt_provider=active["stt"],
            llm_provider=active["llm"],
            tts_provider=active["tts"],
        )

        language_state = LanguageState(
            # FIXED: config.yaml's language.sustained_switch_turns has
            # existed the whole time but nothing ever read it -- the
            # hysteresis threshold was hardcoded (and, before the
            # language_state.py fix, not implemented at all). Now the value
            # you actually set in config.yaml is the value that's used.
            sustained_switch_turns=config.get("language", {}).get(
                "sustained_switch_turns",
                2,
            ),
        )

        shutdown_state = {
            "active": False
        }

        # ----------------------------------------------------
        # LLM tools
        # ----------------------------------------------------
        def _latest_user_utterance() -> str:
            for message in reversed(messages):
                if message.get("role") == "user":
                    content = message.get("content", "")
                    return content.strip() if isinstance(content, str) else ""
            return ""

        async def end_call(
            params: FunctionCallParams,
        ):
            tool_call_started_at = time.monotonic()
            latest_user_text = _latest_user_utterance()

            # Never allow the model to terminate an outbound call before the
            # caller has actually spoken. This prevents an LLM/tool-call
            # hallucination such as end_call("hello") from killing the call.
            # This is the ONE guard we keep here — everything else about
            # *when* it's appropriate to hang up is already carefully
            # specified in `termination_rules` (config.yaml) and is the
            # model's judgement call, not a keyword match.
            #
            # A previous version of this function additionally required the
            # caller's latest utterance to contain one of a hardcoded list
            # of English/Hindi termination phrases ("not interested", "bye",
            # "बस", ...) before honoring the model's end_call tool call. That
            # silently made it impossible for a Telugu or Tamil caller to
            # ever end a call gracefully — no matter how clearly they asked
            # to stop, the keyword gate would reject it and the call would
            # only end via the 8-second forced-hangup fallback below. Rather
            # than try to maintain a translated keyword list per supported
            # language forever (which will always be incomplete), we trust
            # the model's own judgement here, backed by the "no utterance
            # yet" guard above and the very explicit termination_rules
            # prompt.
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
                        run_llm=True
                    ),
                )
                return

            logger.info(
                "[{}] end_call requested language={} latest_user_utterance={!r}",
                stream_id,
                language_state.current_language,
                latest_user_text,
            )

            try:
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

                        # FIXED: this originally called EndTaskFrame(), which
                        # turned out to still exist in the installed Pipecat
                        # version (1.8.1) — deprecated in favor of
                        # EndWorkerFrame, not gone entirely. (My first pass at
                        # this fix incorrectly concluded EndTaskFrame no longer
                        # existed at all, based on a grep that missed it due to
                        # decorator indentation — worth being upfront about,
                        # since getting this right matters more than getting it
                        # fast.) Rather than lean on a deprecated frame's exact
                        # direction/flush semantics, this now calls task.cancel()
                        # directly — the same mechanism already used successfully
                        # at two other backup-termination points in this file
                        # (the hard call-duration timeout). It's also the more
                        # semantically correct choice here regardless: this
                        # fallback only runs when the graceful path has already
                        # failed to complete in 8 seconds, so a forceful cancel
                        # is what's actually wanted, not EndWorkerFrame's
                        # documented "graceful, flush the queue" behavior.
                        await task.cancel()

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
            except Exception:
                # Observability only — record the failure, then re-raise
                # unchanged so error handling/propagation behaves exactly
                # as it did before this metrics call was added.
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

        llm.register_function(
            "end_call",
            end_call,
            cancel_on_interruption=True,
        )

        # ----------------------------------------------------
        # Prompt/context
        # ----------------------------------------------------
        system_prompt = build_system_prompt(
            conversation_call_type,
            config,
            campaign_data,
        )

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]

        llm_tools_raw = (
            config
            .get("providers", {})
            .get("llm", {})
            .get(active["llm"], {})
            .get("params", {})
            .get("tools", [])
        )

        # Language switching is handled locally from STT/transcript signals.
        # Do not expose the model-directed language tool, even if an older
        # config.yaml still contains it.
        llm_tools_raw = [
            raw
            for raw in llm_tools_raw
            if not (
                isinstance(raw, dict)
                and raw.get("function", {}).get("name")
                == "set_conversation_language"
            )
        ]

        llm_tools = [
            FunctionSchema(
                name=raw["function"]["name"],
                description=raw["function"].get(
                    "description",
                    "",
                ),
                properties=raw["function"]
                .get("parameters", {})
                .get("properties", {}),
                required=raw["function"]
                .get("parameters", {})
                .get("required", []),
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
                        VADUserTurnStartStrategy(
                            enable_interruptions=True,
                            enable_user_speaking_frames=True,
                        ),
                        TranscriptionUserTurnStartStrategy(
                            use_interim=True,
                            enable_interruptions=True,
                            enable_user_speaking_frames=False,
                        ),
                    ],
                    stop=[
                        SpeechTimeoutUserTurnStopStrategy(
                            user_speech_timeout=float(
                                turn_config.get(
                                    "user_speech_timeout",
                                    0.5,
                                )
                            ),
                            wait_for_transcript=True,
                        ),
                    ],
                ),
                user_turn_stop_timeout=float(
                    turn_config.get(
                        "user_turn_stop_timeout",
                        2.0,
                    )
                ),
            ),
        )

        language_observer = LanguageObserver(
            stream_id=stream_id,
            language_state=language_state,
            tts=tts,
            tts_provider=active["tts"],
            messages=messages,
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

        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                language_observer,
                context_aggregator.user(),
                llm,
                spoken_text_guard,
                tts,
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

            # Resolve a real customer name where we have one. For the web
            # test harness there is no campaign_data at all, so previously
            # this always fell back to the literal string "there" —
            # producing "Hi, am I speaking with there?" on every dev call.
            # config.yaml already carries a `test_customer_name` for exactly
            # this situation; use it instead of a fallback that only reads
            # correctly when it happens to fill a real name slot.
            customer_name = None
            if campaign_data:
                customer_name = campaign_data.get("customer_name")
            if not customer_name and call_type == "web":
                customer_name = config.get("test_customer_name")

            # Outbound is the conversation contract for both Vobiz outbound
            # calls and the browser/web test harness. Never fall back to the
            # inbound greeting for a web test.
            if conversation_call_type == "outbound":
                if customer_name:
                    greeting_template = config.get(
                        "greeting_outbound",
                        "Hi, am I speaking with {customer_name}?",
                    )
                    greeting = str(greeting_template).replace(
                        "{customer_name}",
                        customer_name,
                    )
                else:
                    # No name available at all (e.g. misconfigured test
                    # harness call). Use a fallback phrased so it's
                    # grammatical without a name slot, instead of
                    # substituting a placeholder word like "there" into a
                    # template built for a name.
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

            # Speak the opening directly, and let Pipecat's own assistant
            # context aggregator record it into `messages` via
            # append_to_context=True. Do NOT also manually append it to
            # `messages` here: a previous version did both, and the
            # aggregator recording the same TTSSpeakFrame text a second
            # time was producing two identical assistant turns in context
            # before the caller ever said a word (visible directly in call
            # logs), inflating every subsequent LLM request for the whole
            # call. There should be exactly one source of truth for what
            # was spoken.
            #
            # Requires a pipecat-ai version that supports
            # TTSSpeakFrame(append_to_context=...). Check `pip show
            # pipecat-ai` if this argument is rejected; upgrading is the
            # right move either way; this exact class of duplicate-context
            # bug has had multiple fixes land upstream.
            await task.queue_frames(
                [
                    TTSSpeakFrame(text=greeting, append_to_context=True),
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
        # FIXED: STREAM_PROVIDER_CALL_IDS accumulated one entry per call,
        # forever — nothing ever removed one. Not a cross-call correctness
        # issue (it's correctly keyed per stream_id), but an unbounded
        # memory leak that would eventually degrade or crash the whole
        # process after enough cumulative call volume, taking every
        # concurrent call down with it, at a time that would be genuinely
        # hard to predict or diagnose.
        STREAM_PROVIDER_CALL_IDS.pop(stream_id, None)

        if call_metrics is not None:
            call_metrics.finalize()

        if not aiohttp_session.closed:
            await aiohttp_session.close()

        logger.remove(
            log_handler
        )
