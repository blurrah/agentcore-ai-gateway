"""examples/05 uses typed params; confirm they render to the same Gateway body shape.

Uses the SDK's internal request serializer, so this test is pinned to the beta
version in uv.lock and may need updating when the SDK changes.
"""

import importlib.util
import os
from pathlib import Path

import ai
from ai.providers.ai_gateway.protocol._shared import merge_extra_body, request_options

os.environ.setdefault("AI_GATEWAY_API_KEY", "test-key")

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "05_agent_ai_sdk_python.py"


def test_example_module_constructs_model_agent_and_tool():
    spec = importlib.util.spec_from_file_location("example_05", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.model.id == "anthropic/claude-sonnet-4.6"
    assert module.model.provider.default_base_url.startswith("https://ai-gateway.vercel.sh/")
    assert module.model.provider.api_key_env == "AI_GATEWAY_API_KEY"
    assert isinstance(module.agent, ai.Agent)
    assert "main" in module.app.handlers


def test_typed_params_render_to_gateway_provider_options():
    params = ai.InferenceRequestParams(
        routing=ai.RoutingParams(
            provider_order=("anthropic", "bedrock", "vertex"),
            fallback_models=("openai/gpt-5.4", "google/gemini-3-flash"),
        ),
        safety_identifier="sess-abc",
        tags=frozenset({"app:agentcore-ai-gateway"}),
        extra_body={
            "providerOptions": {
                "gateway": {
                    "providerTimeouts": {"byok": {"anthropic": 5_000, "bedrock": 10_000}},
                }
            }
        },
    )

    body, _headers, _query = request_options(params, model_id="anthropic/claude-sonnet-4.6")
    merge_extra_body(body, params.extra_body)

    assert body["providerOptions"]["gateway"] == {
        "order": ["anthropic", "bedrock", "vertex"],
        "models": ["openai/gpt-5.4", "google/gemini-3-flash"],
        "user": "sess-abc",
        "tags": ["app:agentcore-ai-gateway"],
        "providerTimeouts": {"byok": {"anthropic": 5_000, "bedrock": 10_000}},
    }
