# app/CustomerSupportAgent/memory/hooks.py
import os
import logging

from bedrock_agentcore.memory import MemoryClient
from strands.hooks import AfterInvocationEvent, HookProvider, HookRegistry, MessageAddedEvent

from .namespaces import build_namespace

logger = logging.getLogger(__name__)


def _text_of(message: dict) -> str:
    """Join the plain-text blocks of a Strands message (a dict)."""
    blocks = message.get("content") or []
    return " ".join(
        b["text"] for b in blocks if isinstance(b, dict) and b.get("text")
    ).strip()


class CustomerMemoryHook(HookProvider):
    """Retrieves customer context before each response, saves the turn after."""

    def __init__(self, actor_id: str, session_id: str, memory_id: str | None = None):
        self.memory_id = memory_id or os.environ.get("MEMORY_CUSTOMERSUPPORTMEMORY_ID")
        if not self.memory_id:
            logger.error("MEMORY_CUSTOMERSUPPORTMEMORY_ID is not set - memory is disabled")
        self.actor_id = actor_id
        self.session_id = session_id
        self.client = MemoryClient()
        self._last_user_text: str | None = None

    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(MessageAddedEvent, self.retrieve_context)
        registry.add_callback(AfterInvocationEvent, self.save_interaction)

    def retrieve_context(self, event: MessageAddedEvent) -> None:
        try:
            message = event.message
            if not self.memory_id or message.get("role") != "user":
                return
            blocks = message.get("content") or []
            text_block = next(
                (b for b in blocks if isinstance(b, dict) and b.get("text")), None
            )
            if text_block is None:
                return  # e.g. a tool-result message, nothing to search on

            user_query = text_block["text"]
            self._last_user_text = user_query  # original text, saved later

            chunks = []
            for key in ("facts", "preferences"):
                try:
                    records = self.client.retrieve_memories(
                        memory_id=self.memory_id,
                        namespace=build_namespace(key, self.actor_id),
                        query=user_query,
                        top_k=3,
                    )
                except Exception:
                    logger.exception("Memory retrieval failed for %s", key)
                    continue
                for record in records:
                    text = (record.get("content") or {}).get("text", "")
                    if text:
                        chunks.append(text)

            logger.info("Retrieved %d memory records for actor %s", len(chunks), self.actor_id)
            if chunks:
                context = "\n".join(f"- {c}" for c in chunks)
                text_block["text"] = f"[Known customer context:\n{context}]\n\n{user_query}"
        except Exception:
            logger.exception("Memory retrieval hook failed")

    def save_interaction(self, event: AfterInvocationEvent) -> None:
        try:
            if not self.memory_id or not self._last_user_text:
                return
            assistant = next(
                (m for m in reversed(event.agent.messages) if m.get("role") == "assistant"),
                None,
            )
            assistant_text = _text_of(assistant) if assistant else ""
            if not assistant_text:
                return
            self.client.create_event(
                memory_id=self.memory_id,
                actor_id=self.actor_id,
                session_id=self.session_id,
                messages=[
                    (self._last_user_text, "USER"),
                    (assistant_text, "ASSISTANT"),
                ],
            )
            logger.info("Saved interaction for actor %s", self.actor_id)
            self._last_user_text = None
        except Exception:
            logger.exception("Memory save failed")