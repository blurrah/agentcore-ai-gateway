"""Offline proof of the Vercel Connect token exchange and AgentCore MCP wiring."""

import importlib.util
import json
import os
from pathlib import Path

import httpx
import pytest
from starlette.testclient import TestClient

os.environ.setdefault("AI_GATEWAY_API_KEY", "gateway-test-key")
os.environ.setdefault("VERCEL_TOKEN", "vercel-test-token")
os.environ.setdefault("VERCEL_CONNECTOR_UID", "mcp.example.com/agentcore")
os.environ.setdefault("MCP_SERVER_URL", "https://mcp.example.com/mcp")

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "09_agent_vercel_connect_mcp.py"
spec = importlib.util.spec_from_file_location("example_09", EXAMPLE)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)

SESSION_HEADER = "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"


@pytest.mark.anyio
async def test_connect_token_exchange_uses_app_subject_and_encoded_connector():
    captured = []

    def transport(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"token": "short-lived-mcp-token"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        token = await example.get_connect_token(client, "mcp.example.com/agentcore", "vercel-secret")

    assert token == "short-lived-mcp-token"
    [request] = captured
    assert str(request.url) == "https://api.vercel.com/v1/connect/token/mcp.example.com%2Fagentcore"
    assert request.headers["authorization"] == "Bearer vercel-secret"
    assert json.loads(request.content) == {"subject": {"type": "app"}}


def test_agentcore_passes_connect_token_to_native_strands_mcp(monkeypatch):
    captured = {}

    async def connect_token(client, connector_uid, vercel_token):
        captured["token_request"] = (connector_uid, vercel_token)
        return "short-lived-mcp-token"

    class FakeMCPClient:
        def __init__(self, **kwargs):
            captured["mcp"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            captured["mcp_closed"] = True

    class FakeAgent:
        def __init__(self, **kwargs):
            captured["agent"] = kwargs

        async def stream_async(self, _prompt):
            yield {"current_tool_use": {"name": "connected_list_issues"}}
            yield {"data": "Done"}

        def cleanup(self):
            captured["agent_cleaned"] = True

    monkeypatch.setattr(example, "get_connect_token", connect_token)
    monkeypatch.setattr(example, "MCPClient", FakeMCPClient)
    monkeypatch.setattr(example, "Agent", FakeAgent)
    monkeypatch.setattr(example, "OpenAIModel", lambda **kwargs: kwargs)

    with TestClient(example.app) as client:
        response = client.post(
            "/invocations",
            json={"prompt": "List my issues"},
            headers={SESSION_HEADER: "session-123"},
        )

    assert response.status_code == 200
    assert captured["token_request"] == ("mcp.example.com/agentcore", "vercel-test-token")
    assert captured["mcp"] == {
        "url": "https://mcp.example.com/mcp",
        "headers": {"Authorization": "Bearer short-lived-mcp-token"},
        "prefix": "connected",
    }
    assert captured["agent_cleaned"] is True
    assert captured["mcp_closed"] is True
    assert "connected_list_issues" in response.text
    assert "Done" in response.text
