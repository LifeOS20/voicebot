from datetime import datetime
from loguru import logger


def _looks_like_orchestration_instruction(text: str) -> bool:
    """Heuristic: does `text` look like a leaked meta/orchestration
    instruction rather than an actual campaign script?

    REVIEWED (your diff, applied): this used to also treat anything under
    500 characters as "looks like an orchestration instruction"
    (`len(t) < 500 or ...`). That's too blunt — plenty of real, legitimate
    campaign_prompt values from the API are short (a one- or two-paragraph
    pitch is well under 500 chars), and were being silently discarded in
    favor of whatever config.yaml's real_estate_sales_script/base_prompt
    fallback says, with no error and no obvious symptom other than "the
    call didn't sound like the campaign we configured." Dropping the
    length gate and relying purely on marker matching is the right call —
    *as long as `markers` below is comprehensive enough to catch actual
    leaks*. Keep your real marker tuple here; the list below is a
    placeholder (I don't have your original list — it was elided as
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
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    voice_rules = str(config.get("voice_rules", "")).strip()

    configured_campaign = str(config.get("real_estate_sales_script", "")).strip()
    configured_prompt = str(config.get("campaign_prompt", "")).strip()
    api_prompt = (
        str(campaign_data.get("campaign_prompt", "")).strip()
        if campaign_data
        else ""
    )

    if call_type == "outbound":
        if api_prompt and not _looks_like_orchestration_instruction(api_prompt):
            campaign_prompt = api_prompt
        elif configured_campaign:
            if api_prompt:
                logger.warning(
                    "Rejected API-provided campaign_prompt as a likely "
                    "orchestration-instruction leak ({} chars); falling "
                    "back to real_estate_sales_script.",
                    len(api_prompt),
                )
            campaign_prompt = configured_campaign
        elif configured_prompt and not _looks_like_orchestration_instruction(
            configured_prompt
        ):
            campaign_prompt = configured_prompt
        else:
            campaign_prompt = str(config.get("base_prompt", "")).strip()
    else:
        campaign_prompt = str(config.get("base_prompt", "")).strip()

    # DO NOT interpolate {customer_name} here - keep prompt static for caching
    # The customer context will be added as a separate message

    parts = [p for p in (voice_rules, campaign_prompt) if p]

    termination_rules = str(config.get("termination_rules", "")).strip()
    if termination_rules:
        parts.append(termination_rules)

    if call_type == "outbound":
        parts.append(
            "OUTBOUND RUNTIME RULE: The application has already spoken the "
            "outbound opening to the caller. Do not generate or repeat the "
            "opening. Never use the inbound greeting \"How can I help you?\". "
            "Respond only to the caller's latest utterance and continue the "
            "conversation naturally from the campaign context."
        )

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
