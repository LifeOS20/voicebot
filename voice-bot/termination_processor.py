"""
TerminationProcessor — a Pipecat FrameProcessor that detects call-ending
phrases in the bot's LLM output via regex and coordinates a clean hangup.

Regex patterns are loaded from config.yaml (termination_processor section)
and compiled once during __init__.
"""

import re
import asyncio

from loguru import logger

from pipecat.frames.frames import (
    TextFrame,
    EndTaskFrame,
    TTSSpeakFrame,
    BotStoppedSpeakingFrame,
    UserStartedSpeakingFrame,
    LLMFullResponseEndFrame,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection


class TerminationProcessor(FrameProcessor):
    def __init__(
        self,
        shutdown_state: dict,
        end_call_pending: dict,
        stream_id: str,
        force_hangup_callback,
        silent_patterns: list,
        spoken_patterns: list,
    ) -> None:
        super().__init__()
        self._shutdown_state = shutdown_state
        self._end_call_pending = end_call_pending
        self._stream_id = stream_id
        self._force_hangup_callback = force_hangup_callback

        self._termination_requested = False
        self._provider_hangup_sent = False
        self._waiting_for_bot_stop = False
        self._hangup_task = None

        # 1. Silent Triggers: Matches should be STRIPPED and trigger hangup
        self.silent_termination_patterns = [
            (entry["name"], re.compile(entry["pattern"], re.IGNORECASE))
            for entry in silent_patterns
        ]

        # 2. Spoken Triggers: Matches should be SPOKEN, then trigger hangup
        self.spoken_termination_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in spoken_patterns
        ]

    async def _hangup_after_bot_stop(self) -> None:
        await asyncio.sleep(3.0)

        if not self._provider_hangup_sent:
            await self._force_hangup_callback("termination_after_bot_stop")
            self._provider_hangup_sent = True

        await self.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)

    async def _schedule_hangup(self, delay: float) -> None:
        await asyncio.sleep(delay)
        if not self._provider_hangup_sent:
            await self._force_hangup_callback("termination_immediate")
            self._provider_hangup_sent = True
        await self.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)

    async def process_frame(self, frame, direction: FrameDirection):
        # Gating: Block all downstream speech if shutdown is active
        if self._shutdown_state["active"] and direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, (TextFrame, TTSSpeakFrame)):
                 return

        await super().process_frame(frame, direction)

        # Handle Interruption: If user speaks during bot's farewell speech, cancel.
        # But if bot already finished speaking (hangup_task running), do NOT cancel.
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, UserStartedSpeakingFrame):
            if self._waiting_for_bot_stop:
                # Bot is still speaking its goodbye — user interrupted mid-speech, cancel
                logger.info(f"[{self._stream_id}] User interrupted mid-goodbye. Cancelling hangup.")
                self._waiting_for_bot_stop = False
                self._termination_requested = False
                self._provider_hangup_sent = False
                self._shutdown_state["active"] = False
                self._end_call_pending["active"] = False
            elif self._hangup_task:
                # Bot already finished speaking goodbye — hangup is committed, ignore
                logger.info(f"[{self._stream_id}] User spoke after goodbye. Hangup is committed, ignoring.")
            await self.push_frame(frame, direction)
            return

        if direction == FrameDirection.UPSTREAM and isinstance(frame, BotStoppedSpeakingFrame):
            if self._waiting_for_bot_stop and not self._hangup_task:
                logger.info(f"[{self._stream_id}] Bot finished speaking. Hanging up in 3.0s.")
                self._hangup_task = asyncio.create_task(self._hangup_after_bot_stop())
                self._waiting_for_bot_stop = False
            # end_call was invoked — bot finished speaking the goodbye, now hang up
            elif self._end_call_pending["active"] and not self._hangup_task:
                logger.info(f"[{self._stream_id}] Bot finished speaking after end_call. Hanging up in 3.0s.")
                self._shutdown_state["active"] = True  # Now gate future output
                self._end_call_pending["active"] = False
                self._hangup_task = asyncio.create_task(self._hangup_after_bot_stop())

            await self.push_frame(frame, direction)
            return

        # End of LLM response marker: passthrough.
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, LLMFullResponseEndFrame):
            await self.push_frame(frame, direction)
            return

        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, (TextFrame, TTSSpeakFrame)):
            # Analyze current frame text for termination triggers, but do not
            # buffer/split normal speech into sentence fragments.
            hangup_requested = False
            cleaned_text = frame.text

            # Check Silent Patterns (Strip them)
            for name, pattern in self.silent_termination_patterns:
                if pattern.search(cleaned_text):
                    hangup_requested = True
                    logger.info(f"[{self._stream_id}] Silent Termination Triggered by pattern: '{name}'")
                    cleaned_text = pattern.sub("", cleaned_text) # Remove the trigger phrase
            
            # Check Spoken Patterns
            if not hangup_requested: # Only check if not already triggered by silent custom commands
                for pattern in self.spoken_termination_patterns:
                    if pattern.search(cleaned_text):
                        hangup_requested = True
                        # Do NOT strip spoken phrases like "Goodbye"
                        break

            if hangup_requested:
                # Strip leading/trailing punctuation and whitespace
                cleaned_text = re.sub(r"^[.,!?\s]+|[.,!?\s]+$", "", cleaned_text)
                
                logger.info(f"[{self._stream_id}] Termination Triggered. Cleaned: '{cleaned_text}'")
                
                # Update state
                self._termination_requested = True
                self._shutdown_state["active"] = True
                
                # Emit any remaining spoken text (e.g. "Goodbye")
                if cleaned_text.strip():
                    self._waiting_for_bot_stop = True
                    if isinstance(frame, TextFrame):
                        await self.push_frame(TextFrame(text=cleaned_text), direction)
                    elif isinstance(frame, TTSSpeakFrame):
                        await self.push_frame(TTSSpeakFrame(text=cleaned_text), direction)
                else:
                    # If silent trigger, hang up immediately (don't wait for speaking end)
                    self._waiting_for_bot_stop = False
                    asyncio.create_task(self._schedule_hangup(delay=0.5))
                return

            # Normal speech: passthrough unchanged to avoid fragmented audio.
            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)
