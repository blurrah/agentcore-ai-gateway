"""AgentCore + Strands MCP tools authenticated by Vercel Connect.

AgentCore runs outside Vercel, so this example authenticates to Connect with a
project-scoped Vercel access token. Connect returns a short-lived MCP bearer
token. The provider credential never becomes an AgentCore environment variable.
"""

import os
from urllib.parse import quote

import httpx
from bedrock_agentcore import BedrockAgentCoreApp
from bedrock_agentcore.runtime.context import RequestContext
from strands import Agent
from strands.models.openai import OpenAIModel
from strands.tools.mcp import MCPClient

app = BedrockAgentCoreApp()


async def get_connect_token(
    client: httpx.AsyncClient,
    connector_uid: str,
    vercel_token: str,
) -> str:
    """Exchange AgentCore's Vercel credential for an app-scoped MCP token."""
    encoded_connector = quote(connector_uid, safe="")
    response = await client.post(
        f"https://api.vercel.com/v1/connect/token/{encoded_connector}",
        headers={"Authorization": f"Bearer {vercel_token}"},
        json={"subject": {"type": "app"}},
    )
    response.raise_for_status()
    token = response.json().get("token")
    if not isinstance(token, str) or not token:
        raise RuntimeError("Vercel Connect returned no token")
    return token


@app.entrypoint
async def handler(payload: dict, context: RequestContext):
    async with httpx.AsyncClient(timeout=20) as client:
        connect_token = await get_connect_token(
            client,
            os.environ["VERCEL_CONNECTOR_UID"],
            os.environ["VERCEL_TOKEN"],
        )

    model = OpenAIModel(
        client_args={
            "api_key": os.environ["AI_GATEWAY_API_KEY"],
            "base_url": "https://ai-gateway.vercel.sh/v1",
        },
        model_id="anthropic/claude-sonnet-4.6",
        params={
            "max_tokens": 1024,
            "extra_body": {
                "providerOptions": {
                    "gateway": {
                        "models": ["openai/gpt-5.4"],
                        "user": context.session_id,
                        "tags": ["app:agentcore-ai-gateway", "tools:vercel-connect-mcp"],
                    }
                }
            },
        },
    )

    mcp_client = MCPClient(
        url=os.environ["MCP_SERVER_URL"],
        headers={"Authorization": f"Bearer {connect_token}"},
        prefix="connected",
    )

    with mcp_client:
        agent = Agent(
            model=model,
            system_prompt="Use the connected MCP tools when they help answer the request.",
            tools=[mcp_client],
            callback_handler=None,
        )
        try:
            async for event in agent.stream_async(payload["prompt"]):
                if "data" in event:
                    yield {"type": "text", "text": event["data"]}
                elif "current_tool_use" in event and event["current_tool_use"].get("name"):
                    yield {"type": "tool_use", "name": event["current_tool_use"]["name"]}
        finally:
            agent.cleanup()


if __name__ == "__main__":
    app.run()
