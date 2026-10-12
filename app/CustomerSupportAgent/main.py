"""
Customer Support AI Agent
=========================
A fully managed AgentCore Runtime application that provides customer support
for an e-commerce platform.  
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import argparse, asyncio, json, logging, os, uuid
from typing import Any, Dict
from typing_extensions import OrderedDict

import boto3
from strands import Agent, tool
from strands.agent.conversation_manager.null_conversation_manager import NullConversationManager
from strands.hooks import (
    AfterInvocationEvent,
    HookProvider,
    HookRegistry,
    MessageAddedEvent,
)
from strands.models import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.streamable_http import streamable_http_client
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.memory import MemoryClient
from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter
from strands_tools.browser import AgentCoreBrowser

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("CSAI_Agent")


# ── TODO 1 — App Initialisation ───────────────────────────────────────────────
# Create a BedrockAgentCoreApp instance.
# This registers the ASGI server for AgentCore deployment.
# There must be exactly one instance per deployment.

app = BedrockAgentCoreApp()

# Suppress interactive tool-consent prompts (required in headless deployments).
os.environ["BYPASS_TOOL_CONSENT"] = "true"


# ── TODO 2 — Configuration ────────────────────────────────────────────────────
# Replace the placeholder strings with your actual AWS resource values.
# You collected these in Part 1 of the INSTRUCTIONS.
#
# GATEWAY_URL format: https://<alias>.gateway.bedrock-agentcore.<region>.amazonaws.com/mcp
# KB_ID       format: 10-character alphanumeric string from the KB console
# REGION:     your AWS region, e.g. "us-east-1"
# MEMORY_ID   format: shown in the AgentCore Memory console

GATEWAY_URL = os.environ.get(
    "AGENTCORE_GATEWAY_CUSTOMERSUPPORTGATEWAY_URL",
    "<gateway_url>",   # Replace with your Gateway URL
)
KB_ID = os.environ.get("KB_ID", "<kbid>")          # Replace with your Knowledge Base ID
REGION = os.environ.get("aws_region", "eu-west-1")  #  Replace with your AWS region
MEMORY_ID = os.environ.get(
    "MEMORY_CUSTOMERSUPPORTMEMORY_ID",
    "<mem_id>",         #  Replace with your Memory ID
)


# ── TODO 3 — Model and Clients ────────────────────────────────────────────────
# Create:
#   1. A BedrockModel using model_id "global.amazon.nova-2-lite-v1:0"
#   2. A MemoryClient with region_name=REGION
#   3. A boto3 client for the "bedrock-agent-runtime" service in REGION

model_id = "global.amazon.nova-2-lite-v1:0"

model = BedrockModel(model_id=model_id)

memory_client = MemoryClient(region_name=REGION)

_bedrock_runtime = boto3.client("bedrock-agent-runtime", region_name=REGION)


# ── TODO 4 — Namespace Helper ─────────────────────────────────────────────────
# Implement get_namespaces() to return a dict mapping strategy type to
# namespace template string.
#
# Steps:
#   1. Call mem_client.get_memory_strategies(memory_id) to get strategy list
#   2. Return a dict: { strategy["type"]: strategy["namespaceTemplates"][0]
#                       for each strategy }
#
# Example output:
#   { "SEMANTIC": "cs_agent/{actorId}/facts",
#     "USER_PREFERENCE": "cs_agent/{actorId}/preferences" }

def get_namespaces(mem_client: MemoryClient, memory_id: str) -> Dict[str, str]:
    """Return a dict mapping strategy type → namespace template string."""
    strategies = mem_client.get_memory_strategies(memory_id=memory_id)
    result: Dict[str, str] = {}
    for strategy in strategies:
        strategy_type = (
            strategy.get("memoryStrategyType") or strategy.get("type", "")
        )
        templates = strategy.get("namespaceTemplates") or strategy.get("namespaces", [])
        if strategy_type and templates:
            result[strategy_type] = templates[0]
    logger.info("Namespaces loaded: %s", result)
    return result


# ── TODO 5 — Memory Hook ──────────────────────────────────────────────────────
# MemoryHook is a HookProvider subclass that adds long-term memory.
#
# __init__: store actor_id, session_id, memory_client, memory_id and
#           call get_namespaces() to populate self.namespaces.
#
# retrieve_customer_context: fires on MessageAddedEvent — queries every
#   strategy namespace and prepends found memories to the user message.
#
# save_support_interaction: fires on AfterInvocationEvent — walks messages
#   backwards to find the last user query and assistant reply, then calls
#   memory_client.create_event().
#
# register_hooks: registers both callbacks on the appropriate events.

class MemoryHook(HookProvider):
    """Long-term memory hook for the customer support agent."""

    def __init__(
        self,
        actor_id: str,
        session_id: str,
        memory_client: MemoryClient,
        memory_id: str,
    ):
        self.actor_id = actor_id
        self.session_id = session_id
        self.memory_client = memory_client
        self.memory_id = memory_id
        self.namespaces = get_namespaces(self.memory_client, self.memory_id)

    def retrieve_customer_context(self, event: MessageAddedEvent) -> None:
        """Retrieve relevant memories and prepend them to the user message."""
        try:
            message = event.message
            if message.get("role") != "user":
                return

            blocks = message.get("content") or []
            text_block = next(
                (b for b in blocks if isinstance(b, dict) and b.get("text")), None
            )
            # Skip tool-result messages — they have no plain text to search on
            if text_block is None:
                return
            if any("toolResult" in b for b in blocks if isinstance(b, dict)):
                return

            user_query = text_block["text"]

            all_context: list[str] = []
            for strategy_type, namespace in self.namespaces.items():
                resolved_namespace = namespace.format(actorId=self.actor_id)
                try:
                    memories = self.memory_client.retrieve_memories(
                        memory_id=self.memory_id,
                        namespace=resolved_namespace,
                        query=user_query,
                        top_k=5,
                    )
                except Exception:
                    logger.exception("Memory retrieval failed for namespace %s", resolved_namespace)
                    continue

                for record in memories:
                    text = (record.get("content") or {}).get("text", "")
                    if text:
                        all_context.append(f"[{strategy_type}] {text}")

            logger.info(
                "Retrieved %d memory records for actor %s", len(all_context), self.actor_id
            )

            if all_context:
                context_block = "Customer Context:\n" + "\n".join(f"- {c}" for c in all_context)
                text_block["text"] = f"{context_block}\n\n{user_query}"

        except Exception:
            logger.exception("retrieve_customer_context hook failed")

    def save_support_interaction(self, event: AfterInvocationEvent) -> None:
        """Save the completed turn to memory after the agent responds."""
        try:
            messages = event.agent.messages

            # Walk backwards to find the last plain-text user message
            customer_query: str = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    for block in msg.get("content") or []:
                        if isinstance(block, dict) and block.get("text"):
                            customer_query = block["text"]
                            break
                    if customer_query:
                        break

            # Walk backwards to find the last assistant response
            agent_response: str = ""
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    for block in msg.get("content") or []:
                        if isinstance(block, dict) and block.get("text"):
                            agent_response = block["text"]
                            break
                    if agent_response:
                        break

            if not customer_query or not agent_response:
                return

            self.memory_client.create_event(
                memory_id=self.memory_id,
                actor_id=self.actor_id,
                session_id=self.session_id,
                messages=[
                    (customer_query, "USER"),
                    (agent_response, "ASSISTANT"),
                ],
            )
            logger.info("Saved interaction for actor %s", self.actor_id)

        except Exception:
            logger.exception("save_support_interaction hook failed")

    def register_hooks(self, registry: HookRegistry) -> None:
        """Register both memory callbacks with the agent hook registry."""
        registry.add_callback(MessageAddedEvent, self.retrieve_customer_context)
        registry.add_callback(AfterInvocationEvent, self.save_support_interaction)


# ── TODO 6 — Knowledge Base Tool ─────────────────────────────────────────────
# Implement search_knowledge_base(query) using the @tool decorator.
#
# Steps:
#   1. Guard: if KB_ID is empty, return a descriptive message
#   2. Call _bedrock_runtime.retrieve(knowledgeBaseId=KB_ID, retrievalQuery=...)
#   3. Extract retrievalResults; return a message if empty
#   4. Join text chunks and return the result

@tool
def search_knowledge_base(query: str) -> str:
    """Search the Amazon product catalog and support knowledge base.

    Use this for product specifications, return policies, warranty
    information, loyalty program details, and order status definitions.

    Args:
        query: The question or topic to search for

    Returns:
        Relevant information retrieved from the knowledge base
    """
    # Guard clause: KB_ID must be configured before calling AWS
    if not KB_ID or not KB_ID.strip():
        return (
            "Knowledge Base is not configured: KB_ID is empty or missing. "
            "Please configure KB_ID before attempting a knowledge-base search."
        )

    resp = _bedrock_runtime.retrieve(
        knowledgeBaseId=KB_ID,
        retrievalQuery={"text": query},
    )

    results = resp.get("retrievalResults", [])
    if not results:
        return f"No information found for: {query}"

    chunks = [r["content"]["text"] for r in results]
    return "\n\n---\n\n".join(chunks)


# ── TODO 7 — Loyalty Discount Tool (Code Interpreter) ────────────────────────
# Implement calculate_loyalty_discount() using the @tool decorator.
#
# The tool:
#   1. Builds a self-contained Python snippet that computes the full
#      discount breakdown (points redemption + tier discount)
#   2. Executes it inside an AgentCore Code Interpreter sandbox
#   3. Falls back to a tier-discount-only calculation if the sandbox
#      is unavailable

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
earn_rates = {{"standard": 1, "device": 2, "fresh": 5}}
tier_rates = {{"Silver": 0.05, "Gold": 0.10, "Platinum": 0.15}}

order_total = {float(order_total)}
loyalty_points = {int(loyalty_points)}
tier = "{tier}"
product_category = "{product_category}"

# Maximum redemption is 50% of the order.
max_points_by_value = int(order_total * 0.5 * 100)

# Points must be redeemed in blocks of 500.
points_redeemed = min(loyalty_points, max_points_by_value)
points_redeemed = (points_redeemed // 500) * 500

points_discount = points_redeemed / 100

subtotal_after_points = order_total - points_discount

tier_discount_pct = tier_rates.get(tier, 0.0)

tier_discount = (
    subtotal_after_points * tier_discount_pct
)

final_total = (
    subtotal_after_points - tier_discount
)

total_savings = (
    order_total - final_total
)

points_earned = int(
    final_total * earn_rates.get(product_category, 1)
)

remaining_points = (
    loyalty_points
    - points_redeemed
    + points_earned
)

import json

result = {{
    "points_redeemed": int(points_redeemed),
    "tier_discount_pct": float(tier_discount_pct),
    "tier_discount": float(round(tier_discount, 2)),
    "final_total": float(round(final_total, 2)),
    "total_savings": float(round(total_savings, 2)),
    "points_earned": int(points_earned),
    "remaining_points": int(remaining_points),
}}

print(json.dumps(result))
"""

    try:
        client = CodeInterpreter(region=REGION)
        client.start()
        try:
            response = client.execute_code(code=code, language="python", clear_context=True)
        finally:
            client.stop()

        # Extract stdout from the streamed response
        if isinstance(response, dict) and "stream" in response:
            for event in response["stream"]:
                result_block = event.get("result", {}) if isinstance(event, dict) else {}
                stdout = result_block.get("structuredContent", {}).get("stdout", "")
                if stdout:
                    return stdout.strip()
                for item in result_block.get("content", []):
                    if item.get("type") == "text" and item.get("text"):
                        return item["text"].strip()

        logger.warning("Could not extract stdout from Code Interpreter response")
        return _fallback_loyalty(loyalty_points, tier, order_total, product_category)

    except Exception as exc:
        logger.warning("Code Interpreter unavailable, using fallback: %s", exc)
        return _fallback_loyalty(loyalty_points, tier, order_total, product_category)


def _fallback_loyalty(loyalty_points: int, tier: str, order_total: float, product_category: str) -> str:
    """Tier-discount-only fallback when the Code Interpreter sandbox is unavailable."""
    tier_rates = {"Silver": 0.05, "Gold": 0.10, "Platinum": 0.15}
    rate = tier_rates.get(tier, 0.00)
    discount = round(order_total * rate, 2)
    return json.dumps({
        "points_redeemed": 0,
        "tier_discount_pct": rate,
        "tier_discount": discount,
        "final_total": round(order_total - discount, 2),
        "total_savings": discount,
        "points_earned": 0,
        "remaining_points": loyalty_points,
        "note": "Fallback calculation — points redemption unavailable (Code Interpreter error).",
    })


# ── TODO 8 — Browse Web Tool ──────────────────────────────────────────────────
# Instantiate AgentCoreBrowser with the AWS region.
# The .browser property exposes the @tool-decorated function used by the agent.

agent_core_browser = AgentCoreBrowser(region=REGION)


# ── TODO 9 — System Prompt ────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a customer support assistant for an e-commerce platform.

You can help customers with:
- Tracking orders and checking delivery status
- Processing refunds and generating return labels
- Answering questions about products, return policies, warranties, and loyalty
  program benefits (search the knowledge base for these)
- Calculating exact loyalty point discounts on an order (always use the
  calculate_loyalty_discount tool — never estimate the math yourself)
- Looking up live information from the web when asked (use browse_web)

Guidelines:
- Always use a tool to look up order, customer, refund, product, or policy
  information rather than guessing — this is real customer data.
- For any discount or points calculation, always call calculate_loyalty_discount.
- If a customer's order or refund can't be found, say so plainly.
- Keep responses concise and friendly.
- If you don't have enough information (e.g. missing an order ID), ask for it.
"""


# ── TODO 10 — Agent Entrypoint ────────────────────────────────────────────────
# Implement the invoke() function decorated with @app.entrypoint.
#
# Steps:
#   1. Extract prompt, actor_id (customer_id), and session_id from payload
#   2. Instantiate MemoryHook for this actor/session
#   3. Build the base tools list: [search_knowledge_base, calculate_loyalty_discount, browse_web]
#   4. Connect to the Gateway via MCPClient, load gateway_tools, extend tools
#   5. Create the Agent with all tools, hooks, model, and system_prompt
#   6. Invoke the agent and stream back the response

@app.entrypoint
async def invoke(payload: dict, context=None):
    """
    Main handler called by AgentCore for every incoming request.

    Expected payload keys:
      prompt      (str, required) — the customer's message
      customer_id (str, optional) — unique customer identifier
      session_id  (str, optional) — session identifier; generated if absent
    """
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")

    prompt = payload.get("prompt", "")
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string")

    actor_id = (
        payload.get("customer_id")
        or payload.get("user_id")
        or getattr(context, "actor_id", None)
        or "default-user"
    )
    session_id = (
        payload.get("session_id")
        or getattr(context, "session_id", None)
        or str(uuid.uuid4())
    )

    # Instantiate the memory hook for this actor/session
    memory_hook = MemoryHook(
        actor_id=actor_id,
        session_id=session_id,
        memory_client=memory_client,
        memory_id=MEMORY_ID,
    )

    # Base inline tools
    tools = [search_knowledge_base, calculate_loyalty_discount, agent_core_browser.browser]

    # Connect to the AgentCore Gateway and load its tools via MCPClient
    if GATEWAY_URL and not GATEWAY_URL.startswith("<"):
        gateway_client = MCPClient(lambda: streamable_http_client(url=GATEWAY_URL))
        with gateway_client:
            try:
                gateway_tools = gateway_client.list_tools_sync()
                tools.extend(gateway_tools)

                logger.info(
                    "Gateway connected successfully. Loaded %d tools.",
                    len(gateway_tools),
                )

            except TimeoutError:
                logger.exception("Gateway tool loading timed out")

            except ConnectionError:
                logger.exception("Gateway connection failed")

            except Exception as exc:
                logger.exception("Gateway tool loading failed: %s", exc)
    else:
        logger.warning("GATEWAY_URL not set — Gateway tools will not be available")

    agent = Agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=tools,
        conversation_manager=NullConversationManager(),
        hooks=[memory_hook],
    )

    async for event in agent.stream_async(prompt):
        if not isinstance(event, dict) or "event" not in event:
            continue
        cbs = event["event"].get("contentBlockStart")
        if cbs is not None and not cbs.get("start"):
            continue
        yield event


# ── CLI entry point ───────────────────────────────────────────────────────────
def main():
    """Run one invocation from the command line for local testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=str)
    args = parser.parse_args()

    async def _run():
        chunks = []
        async for event in invoke(json.loads(args.payload)):
            chunks.append(event)
        return chunks

    result = asyncio.run(_run())
    print(json.dumps(result, default=str))


if __name__ == "__main__":
    app.run()
    # Uncomment the line below and comment app.run() for local CLI testing:
    # main()
