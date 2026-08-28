"""
Builds the system prompt for each call using a two-layer architecture:
  Layer 1: Voice Rules  (always on, compact TTS behaviour constraints)
  Layer 2: Campaign Prompt (business logic, changes per campaign) OR base_prompt fallback

PROMPT-CACHING NOTE:
All genuinely static content (voice_rules, the campaign/base script,
termination_rules) is assembled FIRST, and the one piece of per-call dynamic
content (current time) is appended LAST. This matters because prompt
caching (where supported) works by matching an identical prefix — dynamic
content anywhere in the middle breaks caching for everything that comes
after it. Whether your current LLM provider actually implements prompt
caching wasn't confirmed from its public docs at time of writing; this
ordering costs nothing either way and is the correct default regardless.

The bigger caching-defeating issue is a content one, not code: your sales
script embeds {customer_name} near the very top ("You are calling
{customer_name} regarding..."), which is different for every callee. That
means even the ~250 lines of shared script instructions after it can never
be served from a cached prefix, because the text before them already
differs per call. If you want the bulk of the script to be cacheable across
a campaign, the caller's name needs to move to a short block at the very
end (or into the separate opening greeting message, which already exists),
not be interpolated inline near the top. That's a content/script decision,
not a code one — flagging it here rather than rewriting your script for you.
"""

from datetime import datetime
from loguru import logger


def build_system_prompt(call_type: str, config: dict, campaign_data: dict | None) -> str:
    """
    Construct the full system prompt string for a single call.

    Args:
        call_type:      "inbound" or "outbound"
        config:         The parsed config.yaml dictionary
        campaign_data:  API-provided campaign payload (may be None)

    Returns:
        The assembled system_prompt string ready for the LLM context.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    voice_rules = config.get("voice_rules", "")

    if call_type == "outbound":
        # For outbound: use API-provided campaign prompt -> config campaign_prompt -> base_prompt
        if campaign_data and campaign_data.get("campaign_prompt"):
            campaign_prompt = campaign_data["campaign_prompt"]
        else:
            campaign_prompt = config.get("campaign_prompt", "")

        if campaign_prompt:
            customer_name = campaign_data.get("customer_name", "there") if campaign_data else "there"
            campaign_prompt = campaign_prompt.replace("{customer_name}", customer_name)
            system_prompt = f"{voice_rules}\n\n{campaign_prompt}"
            logger.info(f"Using campaign prompt ({len(campaign_prompt)} chars)")
        else:
            base_prompt = config.get("base_prompt", "You are a helpful AI assistant.")
            system_prompt = f"{voice_rules}\n\n{base_prompt}"
    else:
        # For inbound: always use base_prompt
        base_prompt = config.get("base_prompt", "You are a helpful AI assistant.")
        system_prompt = f"{voice_rules}\n\n{base_prompt}"

    # Append HANGUP instruction for ALL calls
    termination_rules = config.get("termination_rules", "")
    if termination_rules:
        system_prompt += f"\n\n{termination_rules}"
    else:
        # Fallback if config is missing rules (should not happen if config.yaml is updated)
        system_prompt += "\n\nIMPORTANT: When the conversation is naturally over, append ' [HANGUP]' to your final response."

    # Dynamic content goes LAST, after all static content — see module
    # docstring above for why this ordering matters for caching.
    system_prompt += f"\n\nCurrent time: {now}"

    return system_prompt

