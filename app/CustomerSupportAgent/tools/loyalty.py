import json, logging, os

from strands import tool
from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter

REGION = os.environ.get("aws_region", "eu-west-1")

logger = logging.getLogger(__name__)



@tool
def calculate_loyalty_discount(
    loyalty_points: int,
    tier: str,
    order_total: float,
    product_category: str = "standard",
) -> str:
    """
    Calculate the loyalty discount for a customer order using the
    AgentCore Code Interpreter. Runs exact arithmetic in a secure sandbox.

    Args:
        loyalty_points:   Customer's current points balance
        tier:             Customer tier — Silver, Gold, or Platinum
        order_total:      Order total in USD
        product_category: standard, device, or fresh

    Returns:
        Full discount breakdown and final price, as a JSON string
    """
    code = f"""
import json

earn_rates = {{"standard": 1, "device": 2, "fresh": 5}}
tier_rates = {{"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}}

loyalty_points = {loyalty_points}
tier = "{tier}"
order_total = {order_total}
product_category = "{product_category}"

# Redeem points: floor to nearest 500, capped at 50% of order total.
# Assume 100 points = $1 (standard loyalty-program convention)
POINTS_PER_DOLLAR = 100
max_redeemable_value = order_total * 0.5
redeemable_points_by_value = int((max_redeemable_value * POINTS_PER_DOLLAR) // 500) * 500
points_redeemed = min(loyalty_points, redeemable_points_by_value)
points_redeemed = (points_redeemed // 500) * 500

points_value = points_redeemed / POINTS_PER_DOLLAR
subtotal_after_points = order_total - points_value

tier_discount_rate = tier_rates.get(tier, 0.00)
tier_discount = round(subtotal_after_points * tier_discount_rate, 2)

final_total = round(subtotal_after_points - tier_discount, 2)
total_savings = round(order_total - final_total, 2)

earn_rate = earn_rates.get(product_category, 1)
points_earned = int(final_total * earn_rate)
remaining_points = loyalty_points - points_redeemed + points_earned

result = {{
    "order_total": order_total,
    "points_redeemed": points_redeemed,
    "points_value_usd": round(points_value, 2),
    "tier": tier,
    "tier_discount_rate": tier_discount_rate,
    "tier_discount_usd": tier_discount,
    "final_total": final_total,
    "total_savings": total_savings,
    "points_earned": points_earned,
    "remaining_points": remaining_points,
}}

print(json.dumps(result))
"""

    try:
        client = CodeInterpreter(region=REGION)
        client.start()
        try:
            response = client.execute_code(code=code, language="python", clear_context=True)
            stdout_text = _extract_stdout(response)
        finally:
            client.stop()

        if stdout_text:
            return stdout_text.strip()

        logger.warning(f"Could not extract stdout from Code Interpreter response: {response}")
        return _fallback_calculation(loyalty_points, tier, order_total, product_category)

    except Exception as e:
        logger.warning(f"Code Interpreter unavailable, using fallback: {e}")
        return _fallback_calculation(loyalty_points, tier, order_total, product_category)


def _extract_stdout(response: dict) -> str:
    """Extract text output from a Code Interpreter streamed response.

    - `event["result"]["structuredContent"]["stdout"]` holds the printed output;
    - `event["result"]["content"][0]["text"]` mirrors it via MCP-style content blocks.

    Example response structure:

    ```
    {
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": "hello world"
                }
            ],
            "structuredContent": {
            "stdout": "hello world",
            "stderr": "",
            "exitCode": 0,
            "executionTime": 0.0535123348236084
            },
            "isError": false
        }
    }
    ```
    """
    if not isinstance(response, dict) or "stream" not in response:
        return ""

    chunks = []
    for event in response["stream"]:
        result = event.get("result", {}) if isinstance(event, dict) else {}

        # Primary: structuredContent.stdout (confirmed, most direct)
        structured = result.get("structuredContent", {})
        if structured.get("stdout"):
            chunks.append(structured["stdout"])
            continue

        # Fallback: MCP-style content blocks
        for content_item in result.get("content", []):
            if content_item.get("type") == "text":
                chunks.append(content_item.get("text", ""))

    return "\n".join(c for c in chunks if c)


def _fallback_calculation(loyalty_points: int, tier: str, order_total: float, product_category: str) -> str:
    """Tier-discount-only fallback if the Code Interpreter sandbox is unavailable."""
    tier_rates = {"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}
    tier_discount_rate = tier_rates.get(tier, 0.00)
    tier_discount = round(order_total * tier_discount_rate, 2)
    final_total = round(order_total - tier_discount, 2)

    return json.dumps({
        "order_total": order_total,
        "tier": tier,
        "tier_discount_rate": tier_discount_rate,
        "tier_discount_usd": tier_discount,
        "final_total": final_total,
        "note": "Fallback calculation — points redemption unavailable (Code Interpreter error).",
    })