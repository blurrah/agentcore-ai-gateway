"""agent.py again, with the AI SDK for Python (vercel-labs/ai-python) instead of Strands.

AgentCore is framework-agnostic, so the entrypoint is unchanged. What changes:
the SDK routes through AI Gateway by default (no base_url), reads
AI_GATEWAY_API_KEY itself, and exposes fallback models and provider routing as
typed params instead of a raw providerOptions dict. providerTimeouts has no
typed field yet and still goes through extra_body.

Public beta: `uv add ai`.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import ai
from bedrock_agentcore import BedrockAgentCoreApp
from bedrock_agentcore.runtime.context import RequestContext

app = BedrockAgentCoreApp()

model = ai.get_model("anthropic/claude-sonnet-4.6")


@ai.tool
async def current_time(timezone: str = "UTC") -> str:
    """Return the current date and time in an IANA timezone such as Europe/Amsterdam."""
    return datetime.now(ZoneInfo(timezone)).isoformat(timespec="seconds")


agent = ai.Agent(tools=[current_time])


@app.entrypoint
async def handler(payload: dict, context: RequestContext):
    params = ai.InferenceRequestParams(
        routing=ai.RoutingParams(
            provider_order=("anthropic", "bedrock", "vertex"),
            fallback_models=("openai/gpt-5.4", "google/gemini-3-flash"),
        ),
        safety_identifier=context.session_id,
        tags=frozenset({"app:agentcore-ai-gateway"}),
        extra_body={
            "providerOptions": {
                "gateway": {
                    "providerTimeouts": {"byok": {"anthropic": 5_000, "bedrock": 10_000}},
                }
            }
        },
    )

    messages = [
        ai.system_message("You are a concise assistant hosted on Amazon Bedrock AgentCore."),
        ai.user_message(payload["prompt"]),
    ]

    async with agent.run(model, messages, params=params) as stream:
        async for event in stream:
            if isinstance(event, ai.events.TextDelta):
                yield {"type": "text", "text": event.chunk}
            elif isinstance(event, ai.events.ToolStart):
                yield {"type": "tool_use", "name": event.tool_name}


if __name__ == "__main__":
    app.run()
