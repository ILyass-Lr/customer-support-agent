import logging, os
from functools import lru_cache
from bedrock_agentcore.memory import MemoryClient

logger = logging.getLogger(__name__)

MEMORY_ID = os.environ.get("MEMORY_CUSTOMERSUPPORTMEMORY_ID")

_STRATEGY_KEY_MAP = {
    "SEMANTIC": "facts",
    "USER_PREFERENCE": "preferences",
}

@lru_cache(maxsize=1)
def _load_namespace_templates() -> dict:
    """Fetch namespace templates from the deployed Memory resource.

    Calls ``MemoryClient.get_memory_strategies(memory_id)`` which wraps the
    ``GetMemory`` control-plane API and returns a list of normalized strategy
    dicts.  Each dict has the following shape (both old and new field names are
    present after SDK normalization):

    .. code-block:: python

        {
            # Strategy identity (both aliases always present after SDK normalization)
            "strategyId":        str,   # e.g. "abc123"
            "memoryStrategyId":  str,   # same value, newer field name

            # Strategy type (both aliases always present after SDK normalization)
            "type":              str,   # e.g. "SEMANTIC" | "USER_PREFERENCE" | "SUMMARY" | "EPISODIC"
            "memoryStrategyType": str,  # same value, newer field name

            "name":        str,         # e.g. "FactExtractor"
            "description": str,
            "status":      str,         # e.g. "ACTIVE"

            # Namespace templates use camelCase placeholders:
            #   {actorId}          – the actor/user identifier
            #   {sessionId}        – the session identifier
            #   {memoryStrategyId} – the strategy's own ID
            # Example values per strategy type:
            #   SEMANTIC:        ["/support_cases/{actorId}/facts/"]
            #   USER_PREFERENCE: ["/users/{actorId}/preferences/"]
            #   SUMMARY:         ["/summaries/{actorId}/{sessionId}/"]
            #   EPISODIC:        ["/strategy/{memoryStrategyId}/actors/{actorId}/sessions/{sessionId}/"]
            "namespaceTemplates": list[str],

            "configuration": {
                "type": str,            # strategy sub-type / extraction mode
                "extraction":     {...},
                "consolidation":  {...},
                "reflection":     {...},
                "selfManagedConfiguration": {...},
            },
            "memoryRecordSchema": {
                "metadataSchema": [
                    {
                        "key":            str,
                        "type":           str,
                        "extractionType": str,
                        "extractionConfig": {...},
                    }
                ]
            },
            "namespaces":  list[str],   # resolved namespace strings (no placeholders)
            "createdAt":   float,       # epoch seconds
            "updatedAt":   float,
        }

    Returns a dict mapping internal key → first namespace template string:
        {"facts": "/support_cases/{actorId}/facts/",
         "preferences": "/users/{actorId}/preferences/"}
    """
    client = MemoryClient()
    try:
        strategies = client.get_memory_strategies(memory_id=MEMORY_ID)
    except Exception as e:
        logger.error(
            "Failed to load namespace templates from memory ID %s: %s. "
            "Ensure your AWS credentials have bedrock-agentcore:GetMemory "
            "permission and that the session is not using stale cached credentials.",
            MEMORY_ID,
            e,
        )
        raise

    templates = {}
    for strategy in strategies:
        strategy_type = strategy.get("memoryStrategyType") or strategy.get("type")
        key = _STRATEGY_KEY_MAP.get(strategy_type)
        if key and strategy.get("namespaceTemplates"):
            templates[key] = strategy["namespaceTemplates"][0]
    return templates

def build_namespace(template_key: str, actor_id: str) -> str:
    templates = _load_namespace_templates()
    return templates[template_key].format(actorId=actor_id, actor_id=actor_id)