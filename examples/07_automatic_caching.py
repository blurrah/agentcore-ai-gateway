"""Prove Anthropic prompt caching by sending the same long prompt twice.

The second response should report cached input tokens. Cache writes cost more than
normal input, so this is useful for conversations and agent loops, not one-offs.
"""

import os

from openai import OpenAI

client = OpenAI(
    api_key=os.environ["AI_GATEWAY_API_KEY"],
    base_url="https://ai-gateway.vercel.sh/v1",
    default_headers={"x-session-affinity": "automatic-cache-demo"},
)

system_prompt = "\n".join(
    f"Runbook rule {number}: production deployments need two approvals and passing checks."
    for number in range(1, 401)
)
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What is required before a production deployment?"},
]


def send_request():
    return client.chat.completions.create(
        model="anthropic/claude-haiku-4.5",
        messages=messages,
        extra_body={
            "providerOptions": {
                "gateway": {
                    "caching": "auto",
                    "only": ["anthropic"],
                }
            }
        },
    )


def cached_tokens(completion) -> int:
    usage = completion.usage
    details = getattr(usage, "prompt_tokens_details", None)
    return getattr(details, "cached_tokens", 0) or 0


first = send_request()
second = send_request()

print(second.choices[0].message.content)
print("first request cached input tokens :", cached_tokens(first))
print("second request cached input tokens:", cached_tokens(second))
print("cache hit:", cached_tokens(second) > 0)
