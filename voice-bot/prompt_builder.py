from datetime import datetime
from loguru import logger


def _looks_like_orchestration_instruction(text: str) -> bool:
    """Heuristic: does `text` look like a leaked meta/orchestration
    instruction rather than an actual campaign script?

    REVIEWED (your diff, applied): this used to also treat anything under
    500 characters as "looks like an orchestration instruction"
    (`len(t) < 500 or ...`). That's too blunt -- plenty of real, legitimate
    campaign_prompt values from the API are short (a one- or two-paragraph
    pitch is well under 500 chars), and were being silently discarded in
    favor of whatever config.yaml's real_estate_sales_script/base_prompt
    fallback says, with no error and no obvious symptom other than "the
    call didn't sound like the campaign we configured." Dropping the
    length gate and relying purely on marker matching is the right call --
    *as long as `markers` below is comprehensive enough to catch actual
    leaks*. Keep your real marker tuple here; the list below is a
    placeholder (I don't have your original list -- it was elided as
    `(...)` in what you pasted) so this file stays runnable, but it is not
    a substitute for whatever you were actually matching on before.
    """
    t = " ".join((text or "").strip().lower().split())
    if not t:
        return True

    markers = (
        # PLACEHOLDER — replace with your real marker tuple.
        "you are a helpful assistant",
        "you are an ai",
        "as an ai language model",
        "system prompt",
        "ignore previous instructions",
        "you are chatgpt",
        "i am an ai",
        "act as a",
        "roleplay as",
    )

    return any(m in t for m in markers)


_OUTBOUND_RUNTIME_RULE = (
    "OUTBOUND RUNTIME RULE: The application has already spoken the "
    "outbound opening to the caller. Do not generate or repeat the "
    "opening. Never use the inbound greeting \"How can I help you?\". "
    "Respond only to the caller's latest utterance and continue the "
    "conversation naturally from the campaign context."
)

# NEW: inbound calls previously got no equivalent instruction at all.
# The application speaks the inbound greeting via TTSSpeakFrame in
# on_client_connected() before the model ever runs, exactly like it does
# for outbound -- so the model needs the same "don't repeat the greeting"
# guard here that outbound already had. Without this, the model has no
# way of knowing the greeting already happened.
_INBOUND_RUNTIME_RULE = (
    "INBOUND RUNTIME RULE: The application has already greeted the "
    "caller. Do not repeat the greeting or ask \"How can I help you?\" "
    "again. Respond directly to what the caller just said and continue "
    "naturally from there."
)


def build_system_prompt(
    call_type: str,
    config: dict,
    campaign_data: dict | None,
) -> tuple[str, str | None]:
    """
    Returns (system_prompt, customer_context_message).

    The system_prompt is now static (no customer_name interpolation) to enable
    prompt prefix caching. The customer_context_message contains dynamic
    per-call information and is added as a separate context message.

    ARCHITECTURE NOTE (changed): real_estate_sales_script is now the single
    source of property facts for BOTH inbound and outbound calls, instead
    of outbound using real_estate_sales_script while inbound used a
    separate, near-duplicate base_prompt. This fixes two real problems
    found by tracing an actual call log:

    1. config.yaml's `campaign_prompt` key was unreachable dead code --
       real_estate_sales_script always wins over it in the outbound
       priority chain, so it was never once selected in practice. It has
       been dropped from the selection chain entirely (config.yaml no
       longer needs to define it).

    2. base_prompt's OPENING section told the model to say
       "Hi... am I speaking with {customer_name}?" but {customer_name} is
       deliberately never interpolated (see comment below, for caching).
       That meant every inbound call was showing the model a literal,
       unfilled "{customer_name}" placeholder in its instructions. Since
       the app already speaks the greeting in code for both call types,
       that whole OPENING section was dead/harmful instruction anyway.
       base_prompt is now only a defensive last-resort fallback (used
       only if real_estate_sales_script is ever left empty), not the
       primary inbound prompt.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    voice_rules = str(config.get("voice_rules", "")).strip()

    default_script = str(config.get("real_estate_sales_script", "")).strip()
    fallback_script = str(config.get("base_prompt", "")).strip()
    api_prompt = (
        str(campaign_data.get("campaign_prompt", "")).strip()
        if campaign_data
        else ""
    )

    if (
        call_type == "outbound"
        and api_prompt
        and not _looks_like_orchestration_instruction(api_prompt)
    ):
        campaign_prompt = api_prompt
    elif default_script:
        campaign_prompt = default_script
    else:
        if api_prompt:
            logger.warning(
                "Rejected API-provided campaign_prompt as a likely "
                "orchestration-instruction leak ({} chars); falling "
                "back to base_prompt.",
                len(api_prompt),
            )
        campaign_prompt = fallback_script

    # DO NOT interpolate {customer_name} here - keep prompt static for caching
    # The customer context will be added as a separate message

    parts = [p for p in (voice_rules, campaign_prompt) if p]

    termination_rules = str(config.get("termination_rules", "")).strip()
    if termination_rules:
        parts.append(termination_rules)

    if call_type == "outbound":
        parts.append(_OUTBOUND_RUNTIME_RULE)
    else:
        parts.append(_INBOUND_RUNTIME_RULE)

    parts.append(f"Current time: {now}")
    system_prompt = "\n\n".join(parts)

    # Build customer context message for caching-friendly dynamic content
    customer_context = None
    if campaign_data and (customer_name := campaign_data.get("customer_name")):
        customer_context = (
            f"CALL CONTEXT: You are calling {customer_name}. "
            f"Use their name naturally in conversation. "
            f"Do not repeat this context - it is for your reference only."
        )

    logger.info("Using campaign prompt ({} chars)", len(campaign_prompt))
    return system_prompt, customer_context