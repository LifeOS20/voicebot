# Voicebot Audit — Round 2 Addendum

Re-cloned the repo fresh and read every file again, line by line, diffed
against the round-1 review. Real progress since last time: multilingual
support with actual unit tests (8/8 passing), a correctly-wired hard call
timeout, both round-1 termination fixes applied correctly, and a genuinely
smart addition (canceling a pending hangup if the caller interrupts the
goodbye) that wasn't even asked for.

Also one crash-on-boot bug and a silent logging regression that round-1
didn't have. Both fixed and verified below.

---

## 🔴 CRITICAL — FIXED

### 1. `ModuleNotFoundError` on startup — the server could not boot
`main.py` line 19: `from services.chatterbox import ChatterboxTTSService`.
`services/chatterbox.py` had been deleted (correctly — it couldn't scale to
concurrent calls, as flagged in round 1) but the import and a warmup block
referencing it were left behind. Python evaluates imports at module load
time, so this crashed immediately, before a single call could ever connect.
There was also no `chatterbox` entry left in `config.yaml`'s provider
registry, so the warmup block was already fully dead code even before
considering the crash.

**Fixed:** removed the import and the entire warmup block. Verified `main.py`
parses and no other file references Chatterbox anymore.

### 2. Silent logging regression — 13 log calls across `bot.py` were dropping their values
Someone (likely copy-pasting from stdlib-`logging`-style examples) wrote
`logger.warning("[%s] Hard call timeout", stream_id)` in 13 places. Loguru
formats with `{}`, not `%s`. I proved this with an actual loguru run:

```
logger.warning("[%s] Missing Vobiz call ID (%s)", stream_id, trigger)
# -> prints: [%s] Missing Vobiz call ID (%s)     <- value silently dropped
```

This hit exactly the log lines you'd need most during an incident: auth
failures, hangup failures, timeouts, pipeline crashes, init failures. Under
concurrent calls, every one of these lines would look identical
(`[%s] Pipeline error`) with no way to tell which call it belonged to —
directly undermining the "Diagnosable... correlation IDs" principle.

**Fixed:** converted all 13 to `{}` style. Re-verified with a real loguru
run — now correctly prints `[call-abc-123] Missing Vobiz call ID (hard_timeout)`.

---

## 🟠 WARNING — FIXED

### 3. Cost-control config that was never wired up
`config.yaml` had `max_conversation_messages: 20`, and `bot.py` had a
`_prune_history()` function — but nothing ever called it. Every turn of a
long call was re-sending the entire, ever-growing conversation history to
the LLM. A 15-minute call could cost several times more in input tokens by
its end than at its start, silently.

**Fixed:** added a small `_HistoryPruner` frame processor (same pattern as
your existing `LanguageObserver`/`TerminationProcessor`) and wired it into
the pipeline right after the assistant context aggregator. Tested the
pruning logic directly: 61 messages → 20, system prompt preserved, most
recent turn preserved.

### 4. ElevenLabs sample-rate mismatch — still live, still a landmine
Round 1 flagged this; it wasn't touched. `HttpElevenLabsTTSService`
hardcoded `pcm_16000` regardless of what sample rate the call actually
needed. Telephony calls run at 8kHz to match the Vobiz mu-law serializer —
switching to ElevenLabs for a phone call would have produced garbled,
wrong-speed audio. Not your active provider today (Murf is), but still
selectable in config.

**Fixed:** now respects `self.sample_rate` in both the API request and the
emitted frame.

### 5. Cerebras silently missing your entire tool-calling architecture
`sarvam` and `groq` both had the `end_call` / `set_conversation_language`
tools defined; `cerebras` had none. Fail over to it — deliberately or via a
future automatic failover — and the LLM silently loses access to both
tools. No error. Your termination architecture would quietly degrade to
regex-only.

**Fixed:** all three now reference one shared YAML anchor (`&llm_tools` /
`*llm_tools`), so they can't drift apart again — same fix pattern as the
`base_prompt`/`campaign_prompt` drift caught in round 1.

### 6. Frontend still hardcoded to `ws://localhost:8000`
The round-1 fix hadn't made it into the actual repo. Reapplied: the URL is
now derived from `window.location` at runtime.

### 7. `requirements.txt` still carried the removed Chatterbox's dependencies
`torch`, `torchvision`, `torchaudio`, `transformers`, `chatterbox-tts` were
still pinned even though the feature was removed — several GB of dead
weight in every install/Docker image.

**Fixed:** removed. `pipecat-ai` is still unpinned; see note left in the
file — recommend running `pip freeze` once your stack is verified working
end-to-end, since I can't verify a version pin without running your actual
provider stack.

### 8. `body.json` — cleaned, and it wasn't even valid for your current API
Phone numbers were already redacted from round 1, but the file still
carried a real client name (`smartcoin`), business vertical
(`debt_collection_NHR`), and partial personal data (job title, salary,
country) — none of it related to real estate, all of it unnecessary risk in
a public repo. It also didn't match your actual `/outbound` request schema
(`to`, `customer_name`, `campaign_prompt`, `greeting`) at all — it was a
fully orphaned fixture from a different, unrelated product.

**Fixed:** replaced with a minimal, schema-correct example request, entirely
fictional.

### 9. Prompt structure — dynamic content moved to the end
`prompt_builder.py` inserted `Current time: {now}` in the middle of the
system prompt, before the termination rules. Any prompt-caching a provider
might offer requires an identical prefix — dynamic content anywhere in the
middle breaks caching for everything after it.

**Fixed:** reordered so all static content (voice rules, script,
termination rules) comes first, and the one dynamic value comes last.

**Not fixed — needs your call, not mine:** the bigger caching-defeater is
in the script content itself, not the code. `{customer_name}` is
interpolated near the very top of your sales script ("You are calling
{customer_name} regarding..."), which means even the ~250 lines of shared
instructions after it can never share a cached prefix across different
callees in the same campaign — the text already diverges before it gets to
them. Fixing this means moving the name to a short block at the very end
(or relying on the separate opening greeting message, which already
exists) instead of splicing it in near the top. That's a rewrite of your
actual script wording, which you're clearly still iterating on — I didn't
touch it, but the pattern is worth adopting when you're ready.

---

## 🟡 NEW — found via research, not yet fixed (needs your verification)

### 10. Sarvam silently drops `max_completion_tokens`
Confirmed directly from Sarvam's own documentation (via Pipecat's Sarvam
integration docs): `stream_options`, `max_completion_tokens`, and
`service_tier` are on Sarvam's list of unsupported OpenAI parameters,
automatically stripped from requests. Your active LLM provider (`sarvam`)
has `max_completion_tokens: 160` set specifically to cap response length —
for cost, and to keep TTS/spoken duration short. **That cap is not actually
being enforced.** The only thing currently limiting response length is the
soft prompt instruction ("keep ordinary responses to 1–2 short sentences"),
which a model can and occasionally will exceed.

I did **not** patch this blind, because I couldn't confirm from
documentation what parameter (if any) Sarvam does honor for output-length
limiting, and guessing wrong risks breaking generation entirely. Two
concrete paths, both worth testing directly against Sarvam's API before your
demo:

1. Switch the `sarvam` provider's `class_path` from the generic
   `pipecat.services.openai.llm.OpenAILLMService` to Pipecat's **dedicated**
   `pipecat.services.sarvam.llm.SarvamLLMService` — it exists specifically
   to handle Sarvam's API quirks (including this exact unsupported-parameter
   list) and may translate the cap to whatever Sarvam actually accepts.
2. Send a few test requests directly to Sarvam's API with different
   candidate parameter names and confirm empirically which one (if any)
   actually truncates output length.

Also worth confirming empirically: Sarvam's docs do explicitly confirm
"OpenAI-style tool/function calling format" is supported on `sarvam-105b` —
so your core `end_call`/language-switching architecture is on solid,
documented ground. That part I'm confident in.

---

## 🟣 Content/business risk, not code — flagging, not fixing without you

### 11. Your sales script impersonates a real company
`config.yaml`'s script has the agent introduce themselves as calling on
behalf of **Prestige Group** — a real, well-known Bangalore real-estate
developer. If Prestige Group is not literally your client and hasn't
authorized this, having an AI agent call people claiming to represent them
is a real brand/impersonation risk, especially the moment a call recipient
looks the company up. Worth a fictional developer name for any demo where
you don't have that authorization in hand.

### 12. `body.json`, `DESIGN.md`, `ENG.md`, root `SKILL.md` are still in the repo
The PII in `body.json` is now cleaned in this patch, but the repo is still
public — if you haven't already, make it private, and if this was ever
committed with the original PII intact, that data was exposed on the open
internet for however long it was public; treat it as compromised regardless
of the current patch. `DESIGN.md`/`ENG.md`/root `SKILL.md` are still
unrelated third-party Claude Code skill files, not your project's docs —
harmless but worth removing so a future collaborator doesn't mistake them
for real architecture documentation.

---

## ✅ Genuinely good work since round 1 — keep it

1. **`language_state.py`** is well-built: hysteresis (2 consecutive
   reliable turns before auto-switching), confidence gating, an explicit
   override path, and real unit tests. Ran all 8 — they pass, including a
   dedicated call-isolation test, which is exactly the right thing to test
   given round 1's concurrency findings.
2. **The hard call-duration timer is now correctly wired** to
   `config["max_call_duration_seconds"]` and correctly calls the real Vobiz
   hangup — exactly the round-1 fix, applied correctly.
3. **Both termination-processor fixes applied correctly** (end-anchored
   regex matching, distinctly-logged fallback path), plus a genuinely smart
   addition: if the caller interrupts mid-goodbye before the hangup is
   committed, the pending hangup is canceled and state resets cleanly. That
   wasn't something I asked for — it's a real "handle the scenario, don't
   just hang up" instinct, and it's implemented correctly.
4. **The Murf TTS adapter** is solid: proper chunked streaming, correct
   `CancelledError` handling for barge-in, TTFB metrics, and a `finally`
   block that guarantees `TTSStoppedFrame` fires even on error so the
   pipeline never hangs waiting for a stop signal that never comes.
5. **`/outbound`'s fail-closed fix from round 1 was applied correctly.**
