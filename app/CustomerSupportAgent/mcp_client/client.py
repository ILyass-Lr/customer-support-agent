import os
import logging
from mcp.client.streamable_http import streamable_http_client
from strands.tools.mcp.mcp_client import MCPClient

logger = logging.getLogger(__name__)

# CDK injects this automatically based on the gateway name in agentcore.json,
# as AGENTCORE_GATEWAY_{GATEWAY_NAME_UPPERCASE}_URL.
GATEWAY_URL = os.environ.get("AGENTCORE_GATEWAY_CUSTOMERSUPPORTGATEWAY_URL")

def get_streamable_http_mcp_client() -> MCPClient | None:
    """Returns an MCP Client connected to the CustomerSupportGateway."""
    if not GATEWAY_URL:
        logger.warning("GATEWAY_URL not set — falling back to no Gateway tools")
        return None
    return MCPClient(lambda: streamable_http_client(GATEWAY_URL))
