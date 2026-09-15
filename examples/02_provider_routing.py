"""Provider routing: the same model is often served by several upstreams.

Claude, for example, is available from Anthropic's API, Amazon Bedrock and
Google Vertex. By default the Gateway picks based on recent uptime and latency.
These options override that choice.
"""

import os

from openai import OpenAI

client = OpenAI(
    api_key=os.environ["AI_GATEWAY_API_KEY"],
    base_url="https://ai-gateway.vercel.sh/v1",
)

MESSAGES = [{"role": "user", "content": "Reply with the single word: pong"}]


# order: preferred sequence; unlisted providers remain a last resort
completion = client.chat.completions.create(
    model="anthropic/claude-sonnet-4.6",
    messages=MESSAGES,
    extra_body={"providerOptions": {"gateway": {"order": ["bedrock", "anthropic", "vertex"]}}},
)
print("order   :", completion.choices[0].message.content)


# only: hard allow-list
completion = client.chat.completions.create(
    model="anthropic/claude-sonnet-4.6",
    messages=MESSAGES,
    extra_body={"providerOptions": {"gateway": {"only": ["bedrock"]}}},
)
print("only    :", completion.choices[0].message.content)


# sort: rank providers by a live metric ("cost", "ttft" or "tps")
completion = client.chat.completions.create(
    model="anthropic/claude-sonnet-4.6",
    messages=MESSAGES,
    extra_body={"providerOptions": {"gateway": {"sort": "cost"}}},
)
print("sort    :", completion.choices[0].message.content)


# order applies to every model in the fallback chain
completion = client.chat.completions.create(
    model="anthropic/claude-sonnet-4.6",
    messages=MESSAGES,
    extra_body={
        "providerOptions": {
            "gateway": {
                "order": ["bedrock", "anthropic"],
                "models": ["openai/gpt-5.4"],
            }
        }
    },
)
print("combined:", completion.choices[0].message.content)
