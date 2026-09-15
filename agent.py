"""Amazon Bedrock AgentCore agent that calls models through Vercel AI Gateway.

AgentCore Runtime only needs an HTTP server on :8080 with POST /invocations and
GET /ping; BedrockAgentCoreApp provides that. Which model API the agent talks to
is our choice, so instead of the default Bedrock client we give Strands an
OpenAI-compatible client pointed at the Gateway.
"""

import os
from datetime import datetime
from zoneinfo import ZoneInfo

from bedrock_agentcore import BedrockAgentCoreApp
from bedrock_agentcore.runtime.context import RequestContext
from strands import Agent, tool
from strands.models.openai import OpenAIModel

app = BedrockAgentCoreApp()

AI_GATEWAY_API_KEY = os.environ["AI_GATEWAY_API_KEY"]


@tool
def current_time(timezone: str = "UTC") -> str:
    """Return the current date and time in an IANA timezone such as Europe/Amsterdam."""
    return datetime.now(ZoneInfo(timezone)).isoformat(timespec="seconds")


@app.entrypoint
async def handler(payload: dict, context: RequestContext):
    model = OpenAIModel(
        client_args={
            "api_key": AI_GATEWAY_API_KEY,
            "base_url": "https://ai-gateway.vercel.sh/v1",
        },
        model_id="anthropic/claude-sonnet-4.6",
        params={
            "max_tokens": 1024,
            "extra_body": {
                "providerOptions": {
                    "gateway": {
                        # Tried in order if the primary model fails on every provider.
                        "models": ["openai/gpt-5.4", "google/gemini-3-flash"],
                        "user": context.session_id,
                        "tags": ["app:agentcore-ai-gateway"],
                    }
                }
            },
        },
    )

    agent = Agent(
        model=model,
        system_prompt="You are a concise assistant hosted on Amazon Bedrock AgentCore.",
        tools=[current_time],
        callback_handler=None,
    )

    async for event in agent.stream_async(payload["prompt"]):
        if "data" in event:
            yield {"type": "text", "text": event["data"]}
        elif "current_tool_use" in event and event["current_tool_use"].get("name"):
            yield {"type": "tool_use", "name": event["current_tool_use"]["name"]}


if __name__ == "__main__":
    app.run()
