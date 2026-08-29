# Latency & Reliability Review

Two important scoping notes before the findings:

1. **No live call data existed to analyze.** This review is architectural —
   based on the actual pipeline structure, the actual code paths, and known,
   verifiable characteristics of the chosen providers (e.g. Groq's inference
   speed is well-documented; your specific network path to Sarvam/Murf/
   Deepgram is not something I can measure from here). Once you've run real
   calls and filled in `VOICEBOT_BENCHMARK.md`, revisit this document and
   replace "likely" with "confirmed" or "actually not the bottleneck."
2. **This review does not propose new architecture.** Every recommendation
   below is either "fix this specific gap" or "verify this specific
   assumption" — not "add a queue" or "add a cache layer." Where a bigger
   idea might help, it's named and explicitly deferred, not built.

---

## Part 1: Where latency actually goes, structurally

A voice turn has one unavoidable serial chain: **caller stops speaking →
STT finalizes → LLM produces a response → TTS produces audio → caller hears
it.** Nothing can be parallelized here (you can't synthesize speech for text
that doesn't exist yet), so the goal is minimizing each serial link, not
eliminating the chain.

### Where each stack's time is likely spent

| Stage | Stack A (Deepgram + Groq + Sarvam TTS) | Stack B (Sarvam STT + Groq + Murf) |
|---|---|---|
| STT finalization | Deepgram's websocket streaming STT is built for low-latency finalization | Sarvam's STT is also websocket-based; no reason to expect a structural difference here — measure both |
| LLM time-to-first-token | Groq is the same in both stacks — this leg should be near-identical between A and B | Same |
| TTS time-to-first-audio | Sarvam TTS uses `SarvamTTSService`, the WebSocket/`InterruptibleTTSService`-based class (not the HTTP variant) — should stream well | Murf's adapter is HTTP-based with chunked streaming (`services/murf_http.py`) — an HTTP request has more fixed per-call overhead (connection setup, TLS handshake reuse depends on keep-alive) than an already-open WebSocket. This is the one structural asymmetry between the two stacks worth specifically measuring. |

**The one concrete, code-level asymmetry**: Stack A's TTS (Sarvam,
WebSocket-based `InterruptibleTTSService`) and Stack B's TTS (Murf, HTTP
POST + streamed response) are architecturally different transport
mechanisms. A persistent WebSocket connection typically has lower per-turn
overhead than a fresh-ish HTTP request per utterance, *if* Murf's adapter
isn't reusing connections efficiently. Check `services/murf_http.py`'s
`aiohttp_session` handling — the current code does reuse a shared
`aiohttp.ClientSession` per call (good — this was verified while fixing
other issues in this file), so connection reuse is already correct.
Remaining TTFA difference between the two stacks, if any, is more likely
provider-side than architecture-side.

### Known, already-fixed sources of *added* latency that are now gone

- The `ServiceFactory` bug that crashed construction (fixed this session)
  wasn't a latency issue — it was a doesn't-start issue. Worth noting only
  because "why is Stack A slow" and "why does Stack A not work at all" were,
  until this session, the same unanswered question.
- Per-call `SileroVADAnalyzer` instantiation (fixed in a prior round) costs
  roughly 1-2 seconds of one-time call-setup latency, not per-turn latency.
  It's the correct tradeoff (a shared VAD instance risks cross-call state
  corruption — see the isolation review from earlier rounds), but it's
  worth explicitly confirming it doesn't show up as a "slow first response"
  in your benchmark numbers; if it does, a small pre-warmed pool (not a
  shared live instance) is the next lever, not something to build blind now.

### The one latency lever that's free and already partially applied

Moving all dynamic content (the current-time stamp) to the end of the
system prompt, so any prompt-prefix caching a provider might offer isn't
defeated by content in the middle, was already done in a prior round. The
bigger remaining lever — `{customer_name}` appearing near the top of the
sales script, which defeats caching for everything after it — is still
open and documented in `prompt_builder.py`. This affects *cost* more
directly than raw per-turn latency (a cached prefix is typically billed
cheaper, and may or may not be faster depending on the provider), so it's
listed here for completeness but the bigger payoff is on the cost side.

### Latency vs. quality tradeoffs actually available to you

- **`max_completion_tokens: 160`** (both stacks, via Groq) directly trades
  response completeness for speed and cost: a hard-capped response finishes
  generating (and therefore starts playing) sooner than an uncapped one.
  160 tokens is already fairly aggressive for natural speech — if a
  response is getting cut off mid-sentence in testing, that's this knob,
  not a bug.
- **`temperature: 0.2`** (both stacks) trades some conversational
  naturalness for determinism/predictability, particularly around reliable
  tool-calling. This is *not* primarily a latency lever (temperature
  doesn't meaningfully change generation speed), so don't reach for it to
  fix slowness — reach for it only if responses feel robotic and you're
  willing to accept slightly less predictable tool-calling in exchange.
- **STT `mode: "codemix"` (Sarvam) vs. Deepgram `language: "multi"`**: a
  code-switching-aware STT mode does more work per audio chunk than a
  single-language mode, in principle. Whether this is measurably slower is
  an empirical question your benchmark run should answer — don't
  preemptively simplify this to save latency you haven't confirmed you're
  losing; code-switching is a real requirement here, not a nice-to-have.

---

## Part 2: Failure paths — what happens when something breaks

Reviewed every point where STT, LLM, TTS, or an outbound network call
(Vobiz hangup) can fail, and traced what actually happens next. Structural
finding first, since it matters for everything below:

**Confirmed: every call gets its own `PipelineTask` and `PipelineRunner`
instance** (both constructed fresh inside `run_bot()`, never shared). This
means a crash inside one call's pipeline is contained to that call's own
asyncio task graph — it cannot directly corrupt or crash another concurrent
call's pipeline. Combined with the per-call service/VAD instantiation
already in place, object-level call isolation is sound.

### What's already handled well

- **`force_provider_hangup`** (the function that actually ends a PSTN call
  via Vobiz's API) has a 5-second timeout and catches every exception —
  network failure, timeout, bad response — without ever propagating. It
  cannot crash whatever called it.
- **Murf and ElevenLabs TTS adapters** both have explicit request timeouts
  (30s) and catch exceptions inside `run_tts`, always emitting
  `TTSStoppedFrame` in a `finally` block — so a TTS failure doesn't leave
  the pipeline hanging, waiting forever for a stop signal that never comes.
- **The outer `run_bot()` structure** catches exceptions at two levels (the
  pipeline-run level and the call-initialization level) and always reaches
  its `finally` block — closing the aiohttp session, removing the per-call
  log handler, and (as of this session) finalizing call metrics — even on
  a crash.

### What isn't handled, and could leave a call hanging, break it ungracefully, or leave something running that shouldn't be

**1. `force_provider_hangup` failure has no retry, and only logs a warning.**
If the Vobiz DELETE request fails — network blip, Vobiz having a bad
moment, a transient 5xx — the function logs and returns. Nothing tries
again. The internal pipeline still gets torn down (`task.cancel()` still
runs in every caller of this function), but the actual PSTN call may still
be connected and billing on Vobiz's side, with no automatic recovery and no
alert beyond a log line someone would have to be watching for. This is the
single largest gap in the termination path's reliability. **Recommended fix
(not implemented, since retry/backoff timing is a decision worth your
input): one retry after a short delay (e.g. 2 seconds) before giving up,
and log the final failure at `ERROR` level, not `WARNING`, so it's visible
to whatever alerting you eventually put in front of these logs.**

**2. STT/LLM/TTS network calls have inconsistent timeout coverage.**
Murf and ElevenLabs (custom adapters, verified above) have explicit
timeouts. Sarvam's and Deepgram's native Pipecat/vendor-SDK clients were
**not independently verified for their own default timeout behavior** in
this review — that would require reading each vendor SDK's internals, which
wasn't done here to stay in scope. If a vendor SDK's underlying client has
no timeout and a provider's connection genuinely hangs (not errors — hangs)
mid-call, the only thing that eventually ends that call is the 15-minute
hard duration ceiling, which is a very long time for a caller to sit in
dead air. **Recommended action: verify this directly, empirically, by
testing what happens when a provider's network path is blocked (e.g. via a
firewall rule during a test) rather than assuming either way.**

**3. No concurrency cap — a gap that predates this session and remains
open.** There is no admission control anywhere in this codebase: no
semaphore, no active-call counter, no limit on simultaneous WebSocket
connections. Under your own framing ("if 500 people are currently on
calls"), this means the server will keep accepting connections and
constructing full pipelines (STT+LLM+TTS+VAD, each with real provider
connections) until something physically gives out — process memory, file
descriptors, or a provider's own account-level concurrency limit being hit
simultaneously across every active call. When that happens, it doesn't fail
one excess call gracefully; it risks destabilizing the whole process, which
(per the isolation finding above) would take down every call in progress,
not just the one that tipped it over. This is the highest-leverage
reliability fix available and remains unimplemented — deliberately, since
sizing the right limit is a capacity-planning decision that depends on your
actual infrastructure, not something to guess at. **Recommended action:
add a simple `asyncio.Semaphore(MAX_CONCURRENT_CALLS)` acquired at call
start and released in `run_bot()`'s existing `finally` block, with new
connections beyond the limit rejected immediately (a clear "at capacity"
response) rather than accepted and left to fail unpredictably later. Size
`MAX_CONCURRENT_CALLS` from your load test, not a guess.**

**4. `STREAM_PROVIDER_CALL_IDS` (a module-level dict) grew forever — fixed
during this session.**
Flagged in an earlier round: every call added an entry, nothing ever
removed one. Not a cross-call correctness bug — it's correctly keyed per
call — but an unbounded memory leak that would eventually degrade or crash
the whole process (again, taking every concurrent call down with it) after
enough cumulative call volume, at a time that would be genuinely difficult
to predict or diagnose without already knowing to look for this. **Fixed**:
the entry is now removed in the same `finally` block that already handles
per-call teardown (closing the aiohttp session, removing the log handler,
finalizing metrics).

### What this means together

Finding 3 (no concurrency cap) is the one that matters most before your
load test: without it, the failure signature isn't "one call breaks," it's
"the whole process eventually breaks, taking every concurrent call down at
once." A load test run *before* adding a concurrency limit would likely
just reproduce and confirm that failure mode at some concurrency threshold,
rather than tell you anything new about the rest of the system. Finding 4
was the same class of problem and is now fixed; finding 3 is the remaining
one in that category.
