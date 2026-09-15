"""Timeouts: two independent layers.

Gateway side (providerTimeouts): a provider that has not started responding
within N milliseconds is abandoned and the Gateway moves to the next provider
or fallback model. Range 1 000..789 000 ms. For streaming, the timer stops at
the first chunk. Applies to your own BYOK provider keys.

Client side (httpx.Timeout): protects against the Gateway or the network being
slow. `max_retries=0` stops the OpenAI SDK from retrying on its own, so the
Gateway is the only thing doing failover.
"""

import os

import httpx
from openai import APITimeoutError, OpenAI

client = OpenAI(
    api_key=os.environ["AI_GATEWAY_API_KEY"],
    base_url="https://ai-gateway.vercel.sh/v1",
    timeout=httpx.Timeout(connect=5.0, read=20.0, write=10.0, pool=5.0),
    max_retries=0,
)

try:
    stream = client.chat.completions.create(
        model="anthropic/claude-sonnet-4.6",
        messages=[{"role": "user", "content": "Explain a race condition in three short sentences."}],
        stream=True,
        extra_body={
            "providerOptions": {
                "gateway": {
                    "order": ["anthropic", "bedrock", "vertex"],
                    "providerTimeouts": {
                        "byok": {
                            "anthropic": 5_000,
                            "bedrock": 10_000,
                        }
                    },
                    "models": ["openai/gpt-5.4"],
                }
            }
        },
    )

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()

except APITimeoutError:
    # Only the client deadline lands here. A Gateway-side timeout is invisible
    # to the caller; a different provider simply answers.
    print("no response within the client deadline")
