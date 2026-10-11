"""
AgentCore Browser integration.

Uses BrowserClient (boto3) to start a managed remote browser session, then
drives it via raw CDP over WebSocket.  No local Playwright binary is required —
the Playwright driver cannot be executed inside the AgentCore Runtime container
(read-only filesystem), so we speak CDP directly using the `websockets` library
which is already present in the container environment.

Why not AgentCoreBrowser from strands-tools?
  strands-agents/tools#453 — Browser.__init__ calls asyncio.set_event_loop()
  which corrupts anyio's _task_states weakref dict under Python 3.14, crashing
  every subsequent thread dispatch in the ASGI server.  Unfixed as of v0.8.9.

CDP execution runs in a dedicated thread with its own asyncio.run() call, fully
isolated from the ASGI event loop.
"""

import asyncio, json, logging, os, threading, websockets
from bedrock_agentcore.tools.browser_client import BrowserClient
from strands import tool

REGION = os.environ.get("aws_region", "eu-west-1")

logger = logging.getLogger(__name__)

_NAVIGATE_TIMEOUT_S = 30
_CONTENT_MAX_CHARS  = 4000


async def _cdp_session(ws_url: str, headers: dict, url: str) -> dict:
    """Open a CDP WebSocket, navigate to *url*, return title + body text."""
    async with websockets.connect(ws_url, additional_headers=headers) as ws:
        msg_id = 0

        async def send(method: str, params: dict | None = None) -> dict:
            nonlocal msg_id
            msg_id += 1
            await ws.send(json.dumps({"id": msg_id, "method": method, "params": params or {}}))
            while True:
                data = json.loads(await asyncio.wait_for(ws.recv(), timeout=_NAVIGATE_TIMEOUT_S))
                if data.get("id") == msg_id:
                    return data.get("result", {})

        # Discover the first page target in the remote session
        targets = (await send("Target.getTargets")).get("targetInfos", [])
        page_targets = [t for t in targets if t["type"] == "page"]
        if not page_targets:
            raise RuntimeError("No page target found in remote browser session")

        # Attach to the target so we can send page-scoped commands
        resp = await send("Target.attachToTarget", {"targetId": page_targets[0]["targetId"], "flatten": True})
        cdp_session_id = resp.get("sessionId")
        if not cdp_session_id:
            raise RuntimeError("Could not attach to page target")

        async def send_to_page(method: str, params: dict | None = None) -> dict:
            nonlocal msg_id
            msg_id += 1
            await ws.send(json.dumps({"id": msg_id, "sessionId": cdp_session_id, "method": method, "params": params or {}}))
            while True:
                data = json.loads(await asyncio.wait_for(ws.recv(), timeout=_NAVIGATE_TIMEOUT_S))
                if data.get("id") == msg_id:
                    return data.get("result", {})

        # Navigate and wait for the page load event
        await send_to_page("Page.enable")
        await send_to_page("Page.navigate", {"url": url})

        deadline = asyncio.get_event_loop().time() + _NAVIGATE_TIMEOUT_S
        while asyncio.get_event_loop().time() < deadline:
            data = json.loads(await asyncio.wait_for(ws.recv(), timeout=_NAVIGATE_TIMEOUT_S))
            if data.get("sessionId") == cdp_session_id and data.get("method") == "Page.loadEventFired":
                break

        # Extract title and body text via JavaScript evaluation
        def js(expr: str) -> str:
            return asyncio.get_event_loop().run_until_complete(
                send_to_page("Runtime.evaluate", {"expression": expr, "returnByValue": True})
            ).get("result", {}).get("value", "")

        title   = (await send_to_page("Runtime.evaluate", {"expression": "document.title",          "returnByValue": True})).get("result", {}).get("value", "")
        content = (await send_to_page("Runtime.evaluate", {"expression": "document.body.innerText", "returnByValue": True})).get("result", {}).get("value", "")

        return {"title": title, "content": content[:_CONTENT_MAX_CHARS]}


def browse_page(url: str, region: str) -> dict:
    """Start an AgentCore Browser session, navigate to *url*, return page data.

    Runs the async CDP logic in a dedicated thread with its own event loop so
    it never interferes with the ASGI server loop.

    Returns:
        {"title": str, "content": str}  on success
        {"error": str, "url": str}      on failure
    """
    result: dict = {}
    errors: list = []

    def _run():
        client = BrowserClient(region=region)
        try:
            client.start()
            ws_url, headers = client.generate_ws_headers()
            result.update(asyncio.run(_cdp_session(ws_url, headers, url)))
        except Exception as exc:
            errors.append(exc)
        finally:
            try:
                client.stop()
            except Exception:
                logger.warning("browser session stop failed", exc_info=True)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join()

    if errors:
        logger.warning("browse_page failed for %s: %s", url, errors[0])
        return {"error": str(errors[0]), "url": url}

    return result


@tool
def browse_web(url: str, task: str) -> str:
    """Navigate to a URL and perform a browsing task, returning the result as text.

    Use this tool whenever the user asks you to look up live information from
    the web, check a website, or retrieve a page title or content.

    Args:
        url:  The full URL to navigate to (e.g. "https://example.com")
        task: What to extract or do on the page (e.g. "return the page title",
              "find the price of product X", "summarise the main content")

    Returns:
        The requested information extracted from the page, or an error message.
    """
    return json.dumps(browse_page(url, region=REGION))
