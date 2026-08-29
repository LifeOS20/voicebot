"""
TerminationProcessor for Pipecat 1.8.x.

Responsibilities:
- Detect explicit termination commands in generated assistant text.
- Remove internal/tool-leak text when detected.
- Preserve ordinary LLM -> TTS streaming unchanged.
- Never forward a frame twice.
- Coordinate graceful termination after spoken goodbye.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from loguru import logger

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    EndTaskFrame,
    LLMFullResponseEndFrame,
    TextFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import (
    FrameDirection,
    FrameProcessor,
)


class TerminationProcessor(FrameProcessor):
    """
    Pass-through termination safety layer.

    Important:
    Do NOT call super().process_frame() here because FrameProcessor's
    implementation already forwards the frame. This processor needs to inspect
    and explicitly forward each frame exactly once.
    """

    def __init__(
        self,
        shutdown_state: dict[str, bool],
        end_call_pending: dict[str, bool],
        stream_id: str,
        force_hangup_callback: Any,
        silent_patterns: list[dict[str, str]],
        spoken_patterns: list[str],
    ) -> None:
        super().__init__()

        self._shutdown_state = shutdown_state
        self._end_call_pending = end_call_pending
        self._stream_id = stream_id
        self._force_hangup_callback = force_hangup_callback

        self._termination_requested = False
        self._provider_hangup_sent = False
        self._waiting_for_bot_stop = False
        self._hangup_task: asyncio.Task | None = None

        self._silent_termination_patterns = [
            (
                entry["name"],
                re.compile(entry["pattern"], re.IGNORECASE),
            )
            for entry in silent_patterns
        ]

        self._spoken_termination_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in spoken_patterns
        ]

    async def _hangup_after_bot_stop(self) -> None:
        """
        Graceful termination:
        allow downstream audio to finish, then terminate the provider call.
        """
        try:
            await asyncio.sleep(3.0)

            if self._provider_hangup_sent:
                return

            await self._force_provider_hangup(
                "termination_after_bot_stop"
            )
            self._provider_hangup_sent = True

            await self.push_frame(
                EndTaskFrame(),
                FrameDirection.UPSTREAM,
            )

        except asyncio.CancelledError:
            raise

        except Exception:
            logger.exception(
                "[{}] Graceful termination task failed",
                self._stream_id,
            )

    async def _schedule_hangup(self, delay: float) -> None:
        """
        Immediate termination path for silent termination triggers.
        """
        try:
            await asyncio.sleep(delay)

            if not self._provider_hangup_sent:
                await self._force_provider_hangup(
                    "termination_immediate"
                )
                self._provider_hangup_sent = True

            await self.push_frame(
                EndTaskFrame(),
                FrameDirection.UPSTREAM,
            )

        except asyncio.CancelledError:
            raise

        except Exception:
            logger.exception(
                "[{}] Immediate termination task failed",
                self._stream_id,
            )

    async def _forward_once(
        self,
        frame: Any,
        direction: FrameDirection,
    ) -> None:
        """
        Explicitly forward a frame exactly once.
        """
        await self.push_frame(frame, direction)

    async def process_frame(
        self,
        frame: Any,
        direction: FrameDirection,
    ) -> None:
        """
        Inspect relevant frames and forward each frame exactly once.
        """

        # ------------------------------------------------------------
        # User interruption while farewell is being spoken
        # ------------------------------------------------------------
        if (
            direction == FrameDirection.DOWNSTREAM
            and isinstance(frame, UserStartedSpeakingFrame)
        ):
            if self._waiting_for_bot_stop:
                logger.info(
                    "[{}] User interrupted farewell; cancelling pending hangup",
                    self._stream_id,
                )

                self._waiting_for_bot_stop = False
                self._termination_requested = False
                self._shutdown_state["active"] = False
                self._end_call_pending["active"] = False

                if self._hangup_task and not self._hangup_task.done():
                    self._hangup_task.cancel()

                self._hangup_task = None

            elif self._hangup_task:
                logger.info(
                    "[{}] User spoke after farewell; hangup remains committed",
                    self._stream_id,
                )

            await self._forward_once(frame, direction)
            return

        # ------------------------------------------------------------
        # Bot stopped speaking after a graceful termination
        # ------------------------------------------------------------
        if (
            direction == FrameDirection.UPSTREAM
            and isinstance(frame, BotStoppedSpeakingFrame)
        ):
            if self._waiting_for_bot_stop and self._hangup_task is None:
                logger.info(
                    "[{}] Farewell finished; scheduling hangup",
                    self._stream_id,
                )

                self._waiting_for_bot_stop = False
                self._hangup_task = asyncio.create_task(
                    self._hangup_after_bot_stop()
                )

            elif (
                self._end_call_pending["active"]
                and self._hangup_task is None
            ):
                logger.info(
                    "[{}] end_call farewell finished; scheduling hangup",
                    self._stream_id,
                )

                self._shutdown_state["active"] = True
                self._end_call_pending["active"] = False

                self._hangup_task = asyncio.create_task(
                    self._hangup_after_bot_stop()
                )

            await self._forward_once(frame, direction)
            return

        # ------------------------------------------------------------
        # If shutdown is already active, suppress further spoken text.
        # Control frames still pass through.
        # ------------------------------------------------------------
        if (
            direction == FrameDirection.DOWNSTREAM
            and self._shutdown_state["active"]
        ):
            if isinstance(frame, (TextFrame, TTSSpeakFrame)):
                return

        # ------------------------------------------------------------
        # LLM response boundary
        # ------------------------------------------------------------
        if (
            direction == FrameDirection.DOWNSTREAM
            and isinstance(frame, LLMFullResponseEndFrame)
        ):
            await self._forward_once(frame, direction)
            return

        # ------------------------------------------------------------
        # Assistant text inspection
        # ------------------------------------------------------------
        if (
            direction == FrameDirection.DOWNSTREAM
            and isinstance(frame, (TextFrame, TTSSpeakFrame))
        ):
            await self._handle_assistant_text(
                frame,
                direction,
            )
            return

        # ------------------------------------------------------------
        # Everything else passes through unchanged.
        # ------------------------------------------------------------
        await self._forward_once(frame, direction)

    async def _handle_assistant_text(
        self,
        frame: TextFrame | TTSSpeakFrame,
        direction: FrameDirection,
    ) -> None:
        text = frame.text or ""

        if not text.strip():
            await self._forward_once(frame, direction)
            return

        cleaned_text = text
        silent_trigger = False
        termination_requested = False

        # ------------------------------------------------------------
        # Silent termination patterns
        # ------------------------------------------------------------
        for name, pattern in self._silent_termination_patterns:
            if pattern.search(cleaned_text):
                termination_requested = True
                silent_trigger = True

                logger.info(
                    "[{}] Silent termination trigger: {}",
                    self._stream_id,
                    name,
                )

                cleaned_text = pattern.sub(
                    "",
                    cleaned_text,
                )

        # ------------------------------------------------------------
        # Spoken termination patterns
        # ------------------------------------------------------------
        if not termination_requested:
            for pattern in self._spoken_termination_patterns:
                match = pattern.search(cleaned_text)

                if not match:
                    continue

                trailing_text = cleaned_text[match.end():]

                # Only treat it as a termination phrase when effectively
                # at the end of the utterance.
                if len(trailing_text.strip()) <= 3:
                    termination_requested = True
                    break

        # ------------------------------------------------------------
        # Normal assistant speech
        # ------------------------------------------------------------
        if not termination_requested:
            await self._forward_once(
                frame,
                direction,
            )
            return

        # ------------------------------------------------------------
        # Termination speech
        # ------------------------------------------------------------
        cleaned_text = re.sub(
            r"^[\s.,!?]+|[\s.,!?]+$",
            "",
            cleaned_text,
        )

        self._termination_requested = True
        self._shutdown_state["active"] = True

        logger.info(
            "[{}] Termination detected; cleaned assistant text={!r}",
            self._stream_id,
            cleaned_text,
        )

        # Spoken farewell remains spoken.
        if cleaned_text.strip() and not silent_trigger:
            self._waiting_for_bot_stop = True

            if isinstance(frame, TTSSpeakFrame):
                await self._forward_once(
                    TTSSpeakFrame(
                        text=cleaned_text,
                    ),
                    direction,
                )
            else:
                await self._forward_once(
                    TextFrame(
                        text=cleaned_text,
                    ),
                    direction,
                )

            return

        # Silent termination has no audio to wait for.
        self._waiting_for_bot_stop = False

        asyncio.create_task(
            self._schedule_hangup(
                delay=0.5,
            )
        )