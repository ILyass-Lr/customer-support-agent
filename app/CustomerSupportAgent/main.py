import os, json, logging
from typing import Any, Annotated
from typing_extensions import OrderedDict

from strands import Agent
from strands.agent.conversation_manager.null_conversation_manager import NullConversationManager 
from bedrock_agentcore.runtime import BedrockAgentCoreApp

from model.load import load_model
from mcp_client.client import get_streamable_http_mcp_client
from memory.hooks import CustomerMemoryHook
from tools.loyalty import calculate_loyalty_discount
from tools.browse import browse_web



REGION = os.environ.get("aws_region", "eu-west-1")

logger = logging.getLogger(__name__)

app = BedrockAgentCoreApp()
log = app.logger

mcp_clients = [get_streamable_http_mcp_client()]

DEFAULT_SYSTEM_PROMPT = """you are a customer support assistant for an e-commerce platform.

you can help customers with:
- tracking orders and checking delivery status
- processing refunds and generating return labels
- answering questions about products, return policies, warranties, and loyalty program benefits (search the knowledge base for these)
- calculating exact loyalty point discounts on an order (always use the calculate_loyalty_discount tool for this — never estimate the math yourself)
- looking up live information from the web when asked (use browse_web)

guidelines:
- always use a tool to look up order, customer, refund, product, or policy information rather than guessing or relying on general knowledge — this is real customer data and it must be accurate.
- for any discount or points calculation, always call calculate_loyalty_discount. do not compute loyalty math yourself, even approximately.
- if a customer's order or refund can't be found, say so plainly rather than inventing details.
- keep responses concise and friendly. match the customer's tone, and keep their stated preferences (e.g. "keep it brief") in mind for the rest of the conversation.
- if you don't have enough information to help (e.g. missing an order id), ask for it rather than guessing.
"""

# Define a collection of tools used by the model
tools = [calculate_loyalty_discount, browse_web]
_INLINE_FUNCTION_NAMES = {"calculate_loyalty_discount"}

# Add MCP client to tools if available
for mcp_client in mcp_clients:
    if mcp_client:
        tools.append(mcp_client)


def _make_conversation_manager():
    return NullConversationManager()

    
def _extract_actor_id(payload: dict, context) -> str:
    """Resolve the customer/user identity for this request.
    The CLI's `--user-id` flag places this directly in the payload body."""
    if isinstance(payload, dict) and payload.get("user_id"):
        return payload["user_id"]
    return "default-user"

# Reuses one Agent per session_id so each session keeps its own in-process
# conversation history (best-effort; resets on cold start). The cache is bounded
# to 128 sessions with LRU eviction (least-recently-used is dropped and its
# history reset) so a single process serving many sessions cannot leak history
# between them or grow without limit. For durable history, attach a session manager.
def agent_factory():
    cache = OrderedDict()
    def get_or_create_agent(session_id, actor_id):
        cache_key = (session_id, actor_id)
        if cache_key in cache:
            cache.move_to_end(cache_key)
            return cache[cache_key]
        if len(cache) >= 128:
            cache.popitem(last=False)
        cache[cache_key] = Agent(
            model=load_model(),
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            tools=tools,
            conversation_manager=_make_conversation_manager(),
            hooks=[
                CustomerMemoryHook(
                    actor_id=actor_id,
                    session_id=session_id,
                )
            ],
        )
        return cache[cache_key]
    return get_or_create_agent
get_or_create_agent = agent_factory()


def strip_trailing_tool_use(messages: Any) -> list[dict]:
    """Strip toolUse blocks from the tail until the last message has none."""
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")

    messages = list(messages)
    while messages:
        last = messages[-1]
        if not isinstance(last, dict):
            raise ValueError("each message must be an object")
        original_content = last.get("content", [])
        if not isinstance(original_content, list) or not all(isinstance(block, dict) for block in original_content):
            raise ValueError("each message content value must be a list of content blocks")

        content = [block for block in original_content if "toolUse" not in block]
        if len(content) == len(original_content):
            break
        if content:
            messages[-1] = {**last, "content": content}
            break
        messages.pop()

    return messages


def _extract_prompt(payload: dict):
    """Accept validated harness messages, tool results, or a plain prompt string."""
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")
    if "messages" in payload:
        return strip_trailing_tool_use(payload["messages"])
    if "tool_results" in payload:
        tool_results = payload["tool_results"]
        if not isinstance(tool_results, list) or not all(
            isinstance(tool_result, dict) and isinstance(tool_result.get("toolUseId"), str)
            for tool_result in tool_results
        ):
            raise ValueError("tool_results must contain objects with a toolUseId string")
        return [{"role": "user", "content": [{"toolResult": {
            "toolUseId": tr["toolUseId"],
            "status": tr.get("status", "success"),
            "content": tr.get("content", []),
        }} for tr in tool_results]}]
    prompt = payload.get("prompt", "")
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string")
    return prompt


def _has_inline_function_call(messages) -> bool:
    """Return True if messages contains an assistant toolUse for an inline function tool."""
    if not _INLINE_FUNCTION_NAMES or not isinstance(messages, list):
        return False
    for msg in messages:
        if msg.get("role") == "assistant":
            for block in msg.get("content", []):
                if isinstance(block, dict) and block.get("toolUse", {}).get("name") in _INLINE_FUNCTION_NAMES:
                    return True
    return False


def _is_inline_function_call(event: dict) -> bool:
    """Check if a contentBlockStart event is for an inline function tool."""
    if not _INLINE_FUNCTION_NAMES:
        return False
    cbs = event.get("contentBlockStart", {})
    start = cbs.get("start", {})
    tool_use = start.get("toolUse") if isinstance(start, dict) else None
    return tool_use is not None and tool_use.get("name") in _INLINE_FUNCTION_NAMES



@app.entrypoint
async def invoke(payload, context):
    log.info("Invoking Agent.....")
    
    session_id = getattr(context, 'session_id', 'default-session')
    actor_id = _extract_actor_id(payload, context)
    agent = get_or_create_agent(session_id, actor_id)

    prompt = _extract_prompt(payload)


    async for event in agent.stream_async(prompt):
        if not isinstance(event, dict) or "event" not in event:
            continue
        cbs = event["event"].get("contentBlockStart")
        if cbs is not None and not cbs.get("start"):
            continue
        yield event


if __name__ == "__main__":
    app.run()
