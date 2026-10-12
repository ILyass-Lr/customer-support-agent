import os
import logging
import boto3
from strands import tool

REGION = os.environ.get("aws_region", "eu-west-1")
KB_ID = os.environ.get("KB_ID", "")

logger = logging.getLogger(__name__)

_bedrock_runtime = boto3.client("bedrock-agent-runtime", region_name=REGION)


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
