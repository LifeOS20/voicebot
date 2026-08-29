"""Per-call conversational language state.

There is intentionally no language dictionary or regex classifier here.

Normal language detection comes from the multilingual STT provider.
The LLM can explicitly request a language change through the
`set_conversation_language` tool.

The application only owns:
- current language
- confidence
- switch history

This module is created once per call and must never be shared between calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

LanguageCode = Literal["en", "hi", "te", "ta"]

SUPPORTED_LANGUAGES: frozenset[LanguageCode] = frozenset(
    {"en", "hi", "te", "ta"}
)

LANGUAGE_LOCALES: dict[LanguageCode, str] = {
    "en": "en-IN",
    "hi": "hi-IN",
    "te": "te-IN",
    "ta": "ta-IN",
}

MIN_STT_CONFIDENCE = 0.70



@dataclass(frozen=True)
class LanguageSwitchEvent:
    from_language: LanguageCode
    to_language: LanguageCode
    reason: Literal[
        "initial_detection",
        "explicit_request",
        "sustained_change",
    ]
    confidence: float
    turn_index: int


@dataclass
class LanguageState:
    current_language: LanguageCode = "en"
    established: bool = False
    candidate_language: Optional[LanguageCode] = None
    candidate_confidence: float = 0.0
    consecutive_candidate_turns: int = 0
    switch_reason: str = "initial"
    turn_index: int = 0
    switch_history: list[LanguageSwitchEvent] = field(default_factory=list)

    def observe_stt(
        self,
        language: Optional[str],
        confidence: Optional[float] = None,
    ) -> tuple[LanguageCode, bool]:
        """Consume one finalized STT language observation.

        A reliable provider-level language decision is applied immediately.
        We do not wait for two turns because that creates a visible mismatch
        when a caller intentionally switches languages.
        """
        self.turn_index += 1

        detected = normalize_language(language)
        if detected is None:
            return self.current_language, False

        confidence_value = (
            normalize_probability(confidence)
            if confidence is not None
            else 0.90
        )

        if confidence_value is None or confidence_value < MIN_STT_CONFIDENCE:
            return self.current_language, False

        if detected == self.current_language:
            self.established = True
            self._clear_candidate()
            return self.current_language, False

        old = self.current_language
        new = self._switch(
            detected,
            "initial_detection" if not self.established else "sustained_change",
            confidence_value,
        )
        return new, new != old

    def set_explicit(
        self,
        language: str,
        reason: str,
    ) -> tuple[LanguageCode, bool]:
        """Apply an explicit language decision requested by the agent."""
        normalized = normalize_language(language)

        if normalized is None:
            raise ValueError(
                f"Unsupported language: {language!r}"
            )

        old = self.current_language

        if old == normalized:
            self.established = True
            self.switch_reason = "explicit_request"
            self._clear_candidate()
            return old, False

        new = self._switch(
            normalized,
            "explicit_request",
            1.0,
        )
        return new, new != old

    def snapshot(self) -> dict:
        return {
            "current_language": self.current_language,
            "established": self.established,
            "candidate_language": self.candidate_language,
            "candidate_confidence": round(
                self.candidate_confidence,
                3,
            ),
            "consecutive_candidate_turns": (
                self.consecutive_candidate_turns
            ),
            "switch_reason": self.switch_reason,
            "turn_index": self.turn_index,
            "switch_count": len(self.switch_history),
        }

    def _switch(
        self,
        language: LanguageCode,
        reason: Literal[
            "initial_detection",
            "explicit_request",
            "sustained_change",
        ],
        confidence: float,
    ) -> LanguageCode:
        if language not in SUPPORTED_LANGUAGES:
            return self.current_language

        event = LanguageSwitchEvent(
            from_language=self.current_language,
            to_language=language,
            reason=reason,
            confidence=round(confidence, 3),
            turn_index=self.turn_index,
        )

        self.switch_history.append(event)
        self.current_language = language
        self.switch_reason = reason
        self.established = True
        self._clear_candidate()

        return language

    def _clear_candidate(self) -> None:
        self.candidate_language = None
        self.candidate_confidence = 0.0
        self.consecutive_candidate_turns = 0


def normalize_language(
    value: Optional[str],
) -> Optional[LanguageCode]:
    if not value:
        return None

    value = str(value).strip().lower().replace("_", "-")
    base = value.split("-", 1)[0]

    if base in {"en", "eng"}:
        return "en"
    if base in {"hi", "hin"}:
        return "hi"
    if base in {"te", "tel"}:
        return "te"
    if base in {"ta", "tam"}:
        return "ta"

    return None


def normalize_probability(
    value: object,
) -> Optional[float]:
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return None

    if probability > 1:
        probability /= 100.0

    return max(0.0, min(1.0, probability))
