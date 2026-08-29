# Voicebot Benchmark & Observability Reference

This documents what `metrics_collector.py` measures, how to read the logs it
produces, and gives you a results table to fill in when you actually run
calls against Stack A and Stack B. It does not contain benchmark numbers
yet — those require live calls with real API keys, which weren't available
during this review. Treat every blank cell below as a to-do before you
present either stack as "production-ready" rather than "should work."

---

## What's actually being measured, and where it comes from

Every metric below is either a Pipecat-native `MetricsFrame` value
(already computed by Pipecat itself once `enable_metrics`/
`enable_usage_metrics` are on — they already are, in `config.yaml`) or
derived from other frames already flowing through the pipeline. Nothing
here required changing STT/LLM/TTS provider behavior; `CallMetricsCollector`
only observes frames that already exist.

| Category | Metric | Source | Log event name |
|---|---|---|---|
| STT | Time to first (partial) result | Pipecat `TTFBMetricsData` | `ttfb` |
| STT | Final transcript + detected language + confidence | `TranscriptionFrame` | `stt_final_transcript` |
| STT | Endpoint/turn-end latency | `UserStoppedSpeakingFrame` timing | folded into `voice_to_voice_latency` |
| STT | Audio usage (seconds billed) | Pipecat `STTUsageMetricsData` | `stt_usage` |
| LLM | Time to first token | Pipecat `TTFATMetricsData` (`ttfat`) | `llm_time_to_first_token` |
| LLM | Total generation latency | Pipecat `ProcessingMetricsData` | `processing_time` |
| LLM | Input/output token usage | Pipecat `LLMUsageMetricsData` | `llm_token_usage` |
| LLM | Prompt-cache hits (if a provider ever supports it) | `cache_read_input_tokens` field on the same event | `llm_token_usage` |
| LLM | Tool-call success/failure + latency | Explicit calls from `end_call`/`set_conversation_language` handlers in `bot.py` | `tool_call` |
| TTS | Time to first audio | Pipecat `TTFAMetricsData` (`ttfa`) | `tts_time_to_first_audio` |
| TTS | Total synthesis latency | Pipecat `ProcessingMetricsData` | `processing_time` |
| TTS | Character usage | Pipecat `TTSUsageMetricsData` | `tts_usage` |
| Call | Caller-stops-speaking → first bot audio | `UserStoppedSpeakingFrame` → `BotStartedSpeakingFrame` timing | `voice_to_voice_latency` |
| Call | Interruptions/barge-ins | `InterruptionFrame` count | `interruption` |
| Call | Provider errors | `ErrorFrame` | `provider_error` |
| Call | Call duration + per-call totals | Wall-clock + accumulated counters | `call_summary` (one line per call, at teardown) |

**Known gap, documented rather than papered over:** Pipecat's
`TurnMetricsData` (its own turn-detection metric) only populates when using
Pipecat's ML-based smart-turn analyzer. This codebase uses plain
VAD-based endpointing (`SileroVADAnalyzer`) instead, so that metric type
will never appear here — turn-end timing instead comes from
`UserStoppedSpeakingFrame`, which VAD-based endpointing does produce.

**What's deliberately NOT logged:** no API keys, no full transcript text, no
full LLM response text. Only latency numbers, token/character counts,
detected language, confidence scores, and short status/error labels
(truncated to 200 characters where an error message could theoretically
echo back caller content).

## Reading the logs

Every line looks like:

```
METRIC call_id=<stream_id> event=<event_name> stt=<provider> llm=<provider> tts=<provider> key=value key=value ...
```

Grep by `call_id=` to reconstruct one call's full timeline. Grep by
`event=call_summary` across many calls to get aggregate cost/latency trends
without touching per-turn detail. `event=tool_call` with
`success=False` should be rare — if it isn't, your `end_call` reliability
has a real problem (see the reliability review below).

---

## Benchmark results — fill in from real calls

Run at least 10 calls per stack under realistic conditions (real network,
real phone or real browser mic, a mix of short and long conversations,
at least one deliberate interruption per call) before trusting either
column. Use `event=call_summary` lines as your primary source; use the
per-turn events for anything that looks like an outlier.

| Metric | Stack A (Deepgram Nova-3 Multi + Groq + Sarvam Bulbul v3) | Stack B (Sarvam Saaras v3 + Groq + Murf Falcon 2) |
|---|---|---|
| STT: avg time to first result (s) | | |
| STT: avg confidence (en) | | |
| STT: avg confidence (hi) | | |
| STT: avg confidence (te) | | |
| LLM: avg time to first token (s) | | |
| LLM: avg total generation latency (s) | | |
| LLM: avg prompt tokens/turn | | |
| LLM: avg completion tokens/turn | | |
| LLM: end_call tool-call success rate | | |
| LLM: set_conversation_language success rate | | |
| TTS: avg time to first audio (s) | | |
| TTS: avg total synthesis latency (s) | | |
| TTS: avg characters/turn | | |
| Call: avg voice-to-voice latency (s) | | |
| Call: p95 voice-to-voice latency (s) | | |
| Call: interruptions handled cleanly (%) | | |
| Call: provider errors per 100 calls | | |
| Call: avg call duration (s) | | |
| Cost: est. $ per call (STT+LLM+TTS+telephony) | | |
| Cost: est. $ per successful conversion | | |
| Language: Telugu recognized correctly (%) | N/A if Deepgram multi doesn't cover Telugu — verify first | |

**Before filling in Stack A's Telugu row:** confirm with Deepgram directly
whether Telugu is in Nova-3's "multi" language set — this wasn't
confirmable from public documentation during this review (see the
prominent warning already in `config.yaml`'s deepgram entry). If it isn't,
Stack A's Telugu column is not "worse numbers," it's "doesn't work" — a
different kind of result, worth marking as such rather than leaving blank.

---

## How to actually collect this

1. Deploy with `active_providers` set to the stack you're testing.
2. Make real calls — a mix of short (<1 min), medium (2-5 min), and long
   (10+ min) conversations, in each of English/Hindi/Telugu, with at least
   one deliberate mid-sentence interruption per call.
3. After each batch, grep `logs/voicebot_*.log` for `event=call_summary`
   and pull the per-turn detail lines for any call that looks unusual.
4. Compute averages/percentiles from those numbers into the table above.
   This document doesn't do that aggregation for you — it's designed to
   emit clean, greppable lines, not to be a dashboard.
