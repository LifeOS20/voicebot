"""
Per-call observability for the voice pipeline.

WHAT THIS IS: a passive FrameProcessor that observes frames already flowing
through the pipeline and emits structured, per-call log lines. It never
modifies, drops, or delays a frame — every frame that arrives is passed
through unchanged, in the same position in the pipeline it already occupies.
Nothing about pipeline behavior or architecture changes by adding this.

WHY A CUSTOM MODULE, NOT JUST "USE PIPECAT'S METRICS": Pipecat already
computes and emits most of the numbers this file cares about — TTFB, time to
first token, time to first audio, token usage, character usage — as
MetricsFrame data, automatically, once enable_metrics/enable_usage_metrics
are on (they already are, in config.yaml). Nothing currently in this
codebase reads those frames; they've been computed and thrown away this
whole time. This module's main job is to catch them and log them, not to
recompute them. The exceptions (things Pipecat's metrics don't cover, added
here explicitly) are: voice-to-voice latency (caller stops speaking -> first
bot audio), interruption counts, provider errors, STT detected
language/confidence, and tool-call success/failure — none of these are
Pipecat MetricsData types, so they're derived from other existing frames
(UserStoppedSpeakingFrame, BotStartedSpeakingFrame, InterruptionFrame,
ErrorFrame, TranscriptionFrame) or from direct calls out of the tool
handlers in bot.py.

CONCURRENCY: one instance of CallMetricsCollector is created per call (same
pattern already used for LanguageObserver/TerminationProcessor/
_HistoryPruner in bot.py), so there is no shared mutable state across calls
and no locking is needed — each call's numbers live only in that call's own
instance.

WHAT IS DELIBERATELY NOT LOGGED: no API keys, no full transcript text, no
full LLM response text. Only latency numbers, token/character counts,
detected language, confidence scores, and short status/error labels. Where
an error message could theoretically contain user-provided content, it's
truncated to 200 characters.

TurnMetricsData (Pipecat's turn-analyzer metric) is NOT handled here: it
only populates when using Pipecat's ML-based smart-turn analyzer. This
codebase uses plain VAD-based endpointing (SileroVADAnalyzer) instead, so
that metric type will simply never appear on this pipeline's MetricsFrame
data. Turn/endpoint timing here comes from UserStoppedSpeakingFrame instead,
which VAD-based endpointing does produce.
"""

from __future__ import annotations

import time
from typing import Any

from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    ErrorFrame,
    Frame,
    InterruptionFrame,
    MetricsFrame,
    TranscriptionFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import (
    LLMUsageMetricsData,
    ProcessingMetricsData,
    STTUsageMetricsData,
    TTFAMetricsData,
    TTFATMetricsData,
    TTFBMetricsData,
    TTSUsageMetricsData,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class CallMetricsCollector(FrameProcessor):
    """Passive, per-call observability processor. One instance per call."""

    def __init__(
        self,
        *,
        call_id: str,
        stt_provider: str,
        llm_provider: str,
        tts_provider: str,
        language: str = "unknown",
    ) -> None:
        super().__init__()
        self._call_id = call_id
        self._stt_provider = stt_provider
        self._llm_provider = llm_provider
        self._tts_provider = tts_provider
        self._language = language

        self._call_start = time.monotonic()
        self._turn_index = 0
        self._user_stopped_speaking_at: float | None = None
        self._voice_to_voice_logged_this_turn = False

        # Aggregated for the single end-of-call summary line.
        self._totals: dict[str, Any] = {
            "llm_prompt_tokens": 0,
            "llm_completion_tokens": 0,
            "tts_characters": 0,
            "stt_audio_seconds": 0.0,
            "interruptions": 0,
            "provider_errors": 0,
            "tool_calls_ok": 0,
            "tool_calls_failed": 0,
        }

    def _log(self, event: str, **fields: Any) -> None:
        detail = " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)
        logger.info(
            "METRIC call_id={} event={} stt={} llm={} tts={} {}",
            self._call_id,
            event,
            self._stt_provider,
            self._llm_provider,
            self._tts_provider,
            detail,
        )

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, MetricsFrame):
            for metrics_data in frame.data:
                self._handle_metrics_data(metrics_data)

        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._turn_index += 1
            self._user_stopped_speaking_at = time.monotonic()
            self._voice_to_voice_logged_this_turn = False

        elif isinstance(frame, BotStartedSpeakingFrame):
            if (
                self._user_stopped_speaking_at is not None
                and not self._voice_to_voice_logged_this_turn
            ):
                latency_ms = (
                    time.monotonic() - self._user_stopped_speaking_at
                ) * 1000
                self._voice_to_voice_logged_this_turn = True
                self._log(
                    "voice_to_voice_latency",
                    turn=self._turn_index,
                    latency_ms=round(latency_ms, 1),
                )

        elif isinstance(frame, InterruptionFrame):
            self._totals["interruptions"] += 1
            self._log(
                "interruption",
                turn=self._turn_index,
                total_so_far=self._totals["interruptions"],
            )

        elif isinstance(frame, ErrorFrame):
            self._totals["provider_errors"] += 1
            self._log(
                "provider_error",
                turn=self._turn_index,
                fatal=getattr(frame, "fatal", None),
                # Truncated: an error message could in principle echo back
                # user-provided content depending on the provider.
                message=str(getattr(frame, "error", frame))[:200],
            )

        elif isinstance(frame, TranscriptionFrame):
            result = getattr(frame, "result", None)
            confidence = (
                result.get("confidence")
                if isinstance(result, dict)
                else None
            )
            detected_language = getattr(frame, "language", None)
            self._log(
                "stt_final_transcript",
                turn=self._turn_index,
                language=detected_language,
                confidence=confidence,
                # Length only, never the transcript text itself.
                char_count=len(frame.text) if getattr(frame, "text", None) else 0,
            )

        await self.push_frame(frame, direction)

    def _handle_metrics_data(self, metrics_data: Any) -> None:
        processor_name = getattr(metrics_data, "processor", None)
        model_name = getattr(metrics_data, "model", None)

        if isinstance(metrics_data, TTFATMetricsData):
            # LLM: time to first token (ttfat = time to first answer token).
            self._log(
                "llm_time_to_first_token",
                turn=self._turn_index,
                processor=processor_name,
                model=model_name,
                ttfat_s=metrics_data.ttfat,
                ttfb_s=metrics_data.ttfb,
                thinking_time_s=metrics_data.thinking_time,
            )

        elif isinstance(metrics_data, TTFAMetricsData):
            # TTS: time to first audio.
            self._log(
                "tts_time_to_first_audio",
                turn=self._turn_index,
                processor=processor_name,
                model=model_name,
                ttfa_s=metrics_data.ttfa,
                ttfb_s=metrics_data.ttfb,
                leading_silence_s=metrics_data.leading_silence,
            )

        elif isinstance(metrics_data, TTFBMetricsData):
            # Generic time-to-first-byte (e.g. STT's first partial result).
            self._log(
                "ttfb",
                turn=self._turn_index,
                processor=processor_name,
                model=model_name,
                value_s=metrics_data.value,
            )

        elif isinstance(metrics_data, ProcessingMetricsData):
            # Total processing time for the emitting processor (covers both
            # "total generation latency" for the LLM and "total synthesis
            # latency" for TTS — which one it is is identified by processor_name).
            self._log(
                "processing_time",
                turn=self._turn_index,
                processor=processor_name,
                model=model_name,
                value_s=metrics_data.value,
            )

        elif isinstance(metrics_data, LLMUsageMetricsData):
            usage = metrics_data.value
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
            cache_read = getattr(usage, "cache_read_input_tokens", None)
            self._totals["llm_prompt_tokens"] += prompt_tokens or 0
            self._totals["llm_completion_tokens"] += completion_tokens or 0
            self._log(
                "llm_token_usage",
                turn=self._turn_index,
                processor=processor_name,
                model=model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                # If your provider ever supports prompt caching, this is the
                # field that will show it working — nonzero cache_read_input
                # _tokens on later turns means cached content was reused.
                cache_read_input_tokens=cache_read,
            )

        elif isinstance(metrics_data, STTUsageMetricsData):
            # value is an STTUsage object (has .audio_seconds), not a plain
            # float — verified directly against the installed metrics
            # module rather than assumed from the field name alone.
            audio_seconds = getattr(metrics_data.value, "audio_seconds", None)
            self._totals["stt_audio_seconds"] += audio_seconds or 0
            self._log(
                "stt_usage",
                turn=self._turn_index,
                processor=processor_name,
                model=model_name,
                audio_seconds=audio_seconds,
            )

        elif isinstance(metrics_data, TTSUsageMetricsData):
            self._totals["tts_characters"] += metrics_data.value or 0
            self._log(
                "tts_usage",
                turn=self._turn_index,
                processor=processor_name,
                model=model_name,
                characters=metrics_data.value,
            )

    def record_tool_call(
        self,
        name: str,
        success: bool,
        latency_ms: float | None = None,
    ) -> None:
        """Tool-call success/failure isn't a Pipecat metric type, so the
        end_call / set_conversation_language handlers in bot.py call this
        directly rather than it being observed passively."""
        if success:
            self._totals["tool_calls_ok"] += 1
        else:
            self._totals["tool_calls_failed"] += 1
        self._log(
            "tool_call",
            name=name,
            success=success,
            latency_ms=round(latency_ms, 1) if latency_ms is not None else None,
        )

    def finalize(self) -> None:
        """Call once at call teardown to emit the single end-of-call summary
        line. Safe to call even if the call failed before any turns happened."""
        duration_s = time.monotonic() - self._call_start
        self._log(
            "call_summary",
            duration_s=round(duration_s, 2),
            turns=self._turn_index,
            **self._totals,
        )
