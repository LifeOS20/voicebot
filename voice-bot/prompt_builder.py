"""
Builds the system prompt for each call using a two-layer architecture:
  Layer 1: Voice Rules  (always on, compact TTS behaviour constraints)
  Layer 2: Campaign Prompt (business logic, changes per campaign) OR base_prompt fallback
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
            system_prompt = f"{voice_rules}\n\n{campaign_prompt}\n\nCurrent time: {now}"
            logger.info(f"Using campaign prompt ({len(campaign_prompt)} chars)")
        else:
            base_prompt = config.get("base_prompt", "You are a helpful AI assistant.")
            system_prompt = f"{voice_rules}\n\n{base_prompt}\n\nCurrent time: {now}"
    else:
        # For inbound: always use base_prompt
        base_prompt = config.get("base_prompt", "You are a helpful AI assistant.")
        system_prompt = f"{voice_rules}\n\n{base_prompt}\n\nCurrent time: {now}"

    # Append HANGUP instruction for ALL calls
    termination_rules = config.get("termination_rules", "")
    if termination_rules:
        system_prompt += f"\n\n{termination_rules}"
    else:
        # Fallback if config is missing rules (should not happen if config.yaml is updated)
        system_prompt += "\n\nIMPORTANT: When the conversation is naturally over, append ' [HANGUP]' to your final response."

    return system_prompt
