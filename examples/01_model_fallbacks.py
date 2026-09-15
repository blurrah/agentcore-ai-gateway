"""Model fallbacks: if the primary model fails, the Gateway tries the next one.

Failover happens inside the Gateway. Your code makes one request and gets one
answer; `completion.model` tells you which model actually produced it.
"""

import os

from openai import OpenAI

client = OpenAI(
    api_key=os.environ["AI_GATEWAY_API_KEY"],
    base_url="https://ai-gateway.vercel.sh/v1",
)

completion = client.chat.completions.create(
    model="anthropic/claude-sonnet-4.6",
    messages=[{"role": "user", "content": "In one sentence, what does a reverse proxy do?"}],
    extra_body={
        "providerOptions": {
            "gateway": {
                "models": ["openai/gpt-5.4", "google/gemini-3-flash"],
            }
        }
    },
)

print("requested:", "anthropic/claude-sonnet-4.6")
print("served by:", completion.model)
print(completion.choices[0].message.content)
