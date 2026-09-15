"""Call the local agent the same way AgentCore Runtime would.

    uv run invoke.py "What time is it in Tokyo?"
"""

import json
import sys
import uuid

import httpx

prompt = " ".join(sys.argv[1:]) or "What time is it in Amsterdam right now?"

with httpx.stream(
    "POST",
    "http://localhost:8080/invocations",
    json={"prompt": prompt},
    # AgentCore sets this header; the agent forwards it to the Gateway as `user`.
    headers={"X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": str(uuid.uuid4())},
    timeout=120,
) as response:
    for line in response.iter_lines():
        if not line.startswith("data: "):
            continue
        event = json.loads(line.removeprefix("data: "))
        if event.get("type") == "tool_use":
            print(f"\n[tool: {event['name']}]")
        elif event.get("type") == "text":
            print(event["text"], end="", flush=True)
        else:
            print(f"\n[error] {event}", file=sys.stderr)

print()
