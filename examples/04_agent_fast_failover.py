"""The AgentCore agent from agent.py, tuned for a latency-sensitive workload.

Same shape as agent.py. The difference is entirely in how the OpenAIModel is
configured: provider order, Gateway-side provider timeouts, model fallbacks,
and a client-side deadline.
"""

import os

import httpx
from bedrock_agentcore import BedrockAgentCoreApp
from bedrock_agentcore.runtime.context import RequestContext
from strands import Agent
from strands.models.openai import OpenAIModel

app = BedrockAgentCoreApp()


@app.entrypoint
async def handler(payload: dict, context: RequestContext):
    model = OpenAIModel(
        client_args={
            "api_key": os.environ["AI_GATEWAY_API_KEY"],
            "base_url": "https://ai-gateway.vercel.sh/v1",
            "timeout": httpx.Timeout(connect=5.0, read=20.0, write=10.0, pool=5.0),
            "max_retries": 0,
        },
        model_id="anthropic/claude-sonnet-4.6",
        params={
            "max_tokens": 1024,
            "extra_body": {
                "providerOptions": {
                    "gateway": {
                        "order": ["anthropic", "bedrock", "vertex"],
                        "providerTimeouts": {
                            "byok": {"anthropic": 5_000, "bedrock": 10_000},
                        },
                        "models": ["openai/gpt-5.4"],
                        "user": context.session_id,
                        "tags": ["app:agentcore-ai-gateway", "profile:fast-failover"],
                    }
                }
            },
        },
    )

    agent = Agent(model=model, callback_handler=None)

    async for event in agent.stream_async(payload["prompt"]):
        if "data" in event:
            yield {"type": "text", "text": event["data"]}


if __name__ == "__main__":
    app.run()
