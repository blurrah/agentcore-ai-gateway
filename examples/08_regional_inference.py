"""Pin provider inference to the US or EU and verify the resolved region.

If no provider can serve the model in the requested region, AI Gateway fails the
request instead of silently sending it elsewhere.
"""

import argparse
import os

from openai import OpenAI

parser = argparse.ArgumentParser()
parser.add_argument("region", nargs="?", choices=("us", "eu"), default="eu")
args = parser.parse_args()

client = OpenAI(
    api_key=os.environ["AI_GATEWAY_API_KEY"],
    base_url="https://ai-gateway.vercel.sh/v1",
)
completion = client.chat.completions.create(
    model="anthropic/claude-haiku-4.5",
    messages=[{"role": "user", "content": "Reply with the single word: pinned"}],
    extra_body={
        "providerOptions": {
            "gateway": {
                "inferenceRegion": {"scope": "zone", "geoRegion": args.region},
            }
        }
    },
)

message = completion.choices[0].message
metadata = getattr(message, "provider_metadata", None)
if metadata is None:
    metadata = (message.model_extra or {}).get("provider_metadata", {})
routing = metadata["gateway"]["routing"]

resolved_region = None
for model_attempt in routing["modelAttempts"]:
    for provider_attempt in model_attempt["providerAttempts"]:
        endpoint = provider_attempt.get("inferenceEndpoint")
        if provider_attempt.get("success") and endpoint:
            resolved_region = endpoint.get("geoRegion")

print(message.content)
print("provider:", routing["finalProvider"])
print("requested region:", args.region)
print("resolved region :", resolved_region)

if resolved_region != args.region:
    raise RuntimeError(f"Expected inference in {args.region!r}, got {resolved_region!r}")
