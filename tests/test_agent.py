"""Offline proof that agent.py sends the right request to Vercel AI Gateway.

No network: the OpenAI client's HTTP transport is replaced with a mock that
records the outgoing request and replies with a canned streaming completion.
"""

import json
import os

import httpx
import pytest
from starlette.testclient import TestClient
from strands.models.openai import OpenAIModel

os.environ.setdefault("AI_GATEWAY_API_KEY", "test-key")

import agent  # noqa: E402  (needs the env var above at import time)

SESSION_HEADER = "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"


def sse(*chunks: dict) -> bytes:
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
    return (body + "data: [DONE]\n\n").encode()


def canned_completion(model: str) -> bytes:
    base = {"id": "chatcmpl-test", "object": "chat.completion.chunk", "created": 1, "model": model}
    return sse(
        {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello from "}, "finish_reason": None}]},
        {**base, "choices": [{"index": 0, "delta": {"content": "the gateway"}, "finish_reason": None}]},
        {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {**base, "choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9}},
    )


def parse_sse(body: str) -> list[dict]:
    return [json.loads(line.removeprefix("data: ")) for line in body.splitlines() if line.startswith("data: ")]


@pytest.fixture
def captured_requests(monkeypatch) -> list[httpx.Request]:
    captured: list[httpx.Request] = []

    def transport(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        model = json.loads(request.content)["model"]
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=canned_completion(model))

    def offline_openai_model(client_args: dict, **config):
        http_client = httpx.AsyncClient(transport=httpx.MockTransport(transport))
        return OpenAIModel(client_args={**client_args, "http_client": http_client}, **config)

    monkeypatch.setattr(agent, "OpenAIModel", offline_openai_model)
    return captured


def test_ping():
    with TestClient(agent.app) as client:
        response = client.get("/ping")
    assert response.status_code == 200
    assert response.json()["status"] == "Healthy"


def test_invocation_goes_through_the_gateway(captured_requests):
    with TestClient(agent.app) as client:
        response = client.post(
            "/invocations",
            json={"prompt": "Say hello"},
            headers={SESSION_HEADER: "sess-abc"},
        )

    assert response.status_code == 200
    events = parse_sse(response.text)
    assert "".join(e["text"] for e in events if e["type"] == "text") == "Hello from the gateway"

    [request] = captured_requests
    assert str(request.url) == "https://ai-gateway.vercel.sh/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key"

    body = json.loads(request.content)
    assert body["model"] == "anthropic/claude-sonnet-4.6"
    assert body["stream"] is True
    assert body["providerOptions"]["gateway"] == {
        "models": ["openai/gpt-5.4", "google/gemini-3-flash"],
        "user": "sess-abc",
        "tags": ["app:agentcore-ai-gateway"],
    }
    assert [t["function"]["name"] for t in body["tools"]] == ["current_time"]


def test_examples_compile():
    import py_compile
    from pathlib import Path

    for path in sorted(Path(__file__).resolve().parents[1].glob("examples/*.py")):
        py_compile.compile(str(path), doraise=True)
